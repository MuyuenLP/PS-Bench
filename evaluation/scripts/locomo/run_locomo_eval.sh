#!/bin/bash

# Common parameters for all scripts
LIB="memu"
VERSION="default"
WORKERS=5
TOPK=20

echo "Running locomo_ingestion.py..."
CUDA_VISIBLE_DEVICES=0 python locomo_ingestion.py --lib $LIB --version $VERSION --workers $WORKERS
if [ $? -ne 0 ]; then
    echo "Error running locomo_ingestion.py"
    exit 1
fi

echo "Running locomo_search.py..."
CUDA_VISIBLE_DEVICES=0 python locomo_search.py --lib $LIB --version $VERSION --top_k $TOPK --workers $WORKERS
if [ $? -ne 0 ]; then
    echo "Error running locomo_search.py"
    exit 1
fi

echo "Running locomo_responses.py..."
python locomo_responses.py --lib $LIB --version $VERSION
if [ $? -ne 0 ]; then
    echo "Error running locomo_responses.py."
    exit 1
fi

echo "Running locomo_eval.py..."
python locomo_eval.py --lib $LIB --version $VERSION --workers $WORKERS --num_runs 3
if [ $? -ne 0 ]; then
    echo "Error running locomo_eval.py"
    exit 1
fi

echo "Running locomo_metric.py..."
python locomo_metric.py --lib $LIB --version $VERSION
if [ $? -ne 0 ]; then
    echo "Error running locomo_metric.py"
    exit 1
fi

echo "All scripts completed successfully!"
