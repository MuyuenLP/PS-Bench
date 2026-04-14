#!/bin/bash
PERSONA="Tim"
LIB="memos-api-online"  
MEMORY_VERSION="default"
VERSION="default"
WORKERS=2
TOPK=3
API_TIMEOUT=120
BATCH_SIZE=10

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export MEMOS_ROOT=$(readlink -f "$SCRIPT_DIR/../../..")
cd "$MEMOS_ROOT" || exit 1

CLASSIFY_SCRIPT="$MEMOS_ROOT/benchmarking/classify.py"
HISTORY_DATA_PATH="$MEMOS_ROOT/benchmarking/data/processed/LoCoMo_ori/${PERSONA}.json"
HARMFUL_DATA_DIR="$MEMOS_ROOT/benchmarking/data/processed/Harmful_Query_Set"
FILE_LIST=(Hate_Speech Self_Harm Abuse Unethical_Behavior Privacy_Leakage Financial_Crime Medical_Crime_and_Advice Sexual_Crime_and_Explicit)

RESULTS_BASE_DIR="$MEMOS_ROOT/evaluation/results/safety_tests_memos/${LIB}_${TOPK}/$PERSONA"
mkdir -p "$RESULTS_BASE_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 1. Memory Ingestion
log "Step 1: Memory Ingestion..."
CUDA_VISIBLE_DEVICES=0 python evaluation/scripts/locomo/locomo_ingestion_safety.py \
    --lib "$LIB" \
    --version "$MEMORY_VERSION" \
    --workers $WORKERS \
    --history_data_path "$HISTORY_DATA_PATH"

for FILE in "${FILE_LIST[@]}"; do
    HARMFUL_FILE="$HARMFUL_DATA_DIR/$FILE.json"
    [ ! -f "$HARMFUL_FILE" ] && continue
    
    FILE_RESULTS_DIR="$RESULTS_BASE_DIR/$FILE"
    mkdir -p "$FILE_RESULTS_DIR"
    FILE_VERSION="${VERSION}_${FILE}"

    # 2. Memory Search
    log "Step 2: Searching - $FILE"
    CUDA_VISIBLE_DEVICES=0 python evaluation/scripts/locomo/locomo_search_safety.py \
        --lib "$LIB" \
        --version "$FILE_VERSION" \
        --memory_version "$MEMORY_VERSION" \
        --top_k $TOPK \
        --workers $WORKERS \
        --qa_data_path "$HARMFUL_FILE" \
        --history_data_path "$HISTORY_DATA_PATH"

    # 3. Generating Responses
    log "Step 3: Generating - $FILE"
    python evaluation/scripts/locomo/locomo_responses_safety.py \
        --lib "$LIB" \
        --version "$FILE_VERSION" \
        --workers $WORKERS \
        --timeout $API_TIMEOUT \
        --batch_size $BATCH_SIZE

    # 4. Classification
    TARGET_RESPONSE_FILE="$MEMOS_ROOT/results/locomo/${LIB}-${FILE_VERSION}/${LIB}_locomo_responses_${CHAT_MODEL}.json"
    [ ! -f "$TARGET_RESPONSE_FILE" ] && continue

    TEMP_RESPONSE_FILE="$FILE_RESULTS_DIR/responses_for_classify.json"
    
    # 提取数据用于分类
    python -c "
import json
with open('$TARGET_RESPONSE_FILE', 'r') as f:
    data = json.load(f)
responses = data.get('locomo_exp_user_0', [])
if responses:
    with open('$TEMP_RESPONSE_FILE', 'w') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
"
    
    log "Step 4: Classifying - $FILE"
    python "$CLASSIFY_SCRIPT" \
        --data_path "$TEMP_RESPONSE_FILE" \
        --output_dir "$FILE_RESULTS_DIR/classify_output"
done

log "Process finished at Classification stage."