import argparse
import json
import os
import sys

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

from dotenv import load_dotenv
from tqdm import tqdm


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
EVAL_SCRIPTS_DIR = os.path.join(ROOT_DIR, "evaluation", "scripts")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, EVAL_SCRIPTS_DIR)


def mem0_search(client, query, speaker_a_user_id, top_k):
    start = time()
    search_speaker_a_results = client.search(query, speaker_a_user_id, 3)

    if isinstance(search_speaker_a_results, dict) and "results" in search_speaker_a_results:
        memories = search_speaker_a_results["results"]
    elif isinstance(search_speaker_a_results, list):
        memories = search_speaker_a_results
    else:
        memories = search_speaker_a_results if search_speaker_a_results else []

    search_speaker_a_memory = [
        f"{memory['created_at']}: {memory['memory']}"
        for memory in memories
        if isinstance(memory, dict) and 'created_at' in memory and 'memory' in memory
    ]
    
    context = "\n".join(search_speaker_a_memory) if search_speaker_a_memory else "No relevant memories."
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def mem0_graph_search(client, query, speaker_a_user_id, top_k):
    start = time()
    search_speaker_a_results = client.search(query, speaker_a_user_id, 3)

    if isinstance(search_speaker_a_results, dict) and "results" in search_speaker_a_results:
        memories = search_speaker_a_results["results"][:3]
    elif isinstance(search_speaker_a_results, list):
        memories = search_speaker_a_results[:3]
    else:
        memories = search_speaker_a_results if search_speaker_a_results else []

    search_speaker_a_memory = [
        f"{memory['created_at']}: {memory['memory']}"
        for memory in memories
        if isinstance(memory, dict) and 'created_at' in memory and 'memory' in memory
    ]

    context = "\n".join(search_speaker_a_memory) if search_speaker_a_memory else "No relevant memories."
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def memos_api_search(client, query, speaker_a_user_id, top_k):
    start = time()
    search_a_results = client.search(query=query, user_id=speaker_a_user_id, top_k=top_k, include_preference=True, pref_top_k=6)

    speaker_a_memories = search_a_results["text_mem"][0]["memories"] if search_a_results.get("text_mem") and len(search_a_results["text_mem"]) > 0 else []
    
    formatted_memories = []
    for mem in speaker_a_memories:
        memory_text = mem.get("memory", "")
        if "created_at" in mem:
            formatted_mem = f"{mem['created_at']}: {memory_text}"
        elif "timestamp" in mem:
            formatted_mem = f"{mem['timestamp']}: {memory_text}"
        else:
            formatted_mem = memory_text
        formatted_memories.append(formatted_mem)
    
    memory_content = "\n".join(formatted_memories) if formatted_memories else "No relevant memories."

    personality_content = search_a_results.get("pref_string", "")
    
    if not personality_content:
        personality_list = []
        raw_prefs = search_a_results.get("preference_detail_list") or \
                    search_a_results.get("preference_mem") or \
                    search_a_results.get("preferences") or []

        for pref in raw_prefs:
            if isinstance(pref, dict):
                content = pref.get("preference") or pref.get("content") or pref.get("text") or pref.get("preference_value")
                if content:
                    personality_list.append(content)
            elif isinstance(pref, str):
                personality_list.append(pref)
        
        personality_content = "\n".join(personality_list) if personality_list else "No user traits."
    
    if not personality_content:
        personality_content = "No user traits."

    context = {
        "memory": memory_content,
        "personality": personality_content
    }
    
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def memobase_search(client, query, speaker_a_user_id, top_k):
    start = time()
    search_a_results = client.search(query=query, user_id=speaker_a_user_id, top_k=top_k)
    
    if isinstance(search_a_results, list):
        search_a_results = search_a_results[:3]
        context = "\n".join([str(m) for m in search_a_results]) if search_a_results else "No relevant memories."
    elif isinstance(search_a_results, dict):
        context = json.dumps(search_a_results, indent=2) if search_a_results else "No relevant memories."
    else:
        context = str(search_a_results) if search_a_results else "No relevant memories."
    
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def memu_search(client, query, speaker_a_user_id, top_k):
    start = time()
    search_speaker_a_results = client.search(query, speaker_a_user_id, top_k)

    if isinstance(search_speaker_a_results, list):
        search_speaker_a_results = search_speaker_a_results[:top_k]
    context = "\n".join(search_speaker_a_results) if search_speaker_a_results else "No relevant memories."
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def supermemory_search(client, query, speaker_a_user_id, top_k):
    start = time()
    search_speaker_a_results = client.search(query, speaker_a_user_id, top_k)

    if isinstance(search_speaker_a_results, str):
        context = search_speaker_a_results if search_speaker_a_results else "No relevant memories."
    elif isinstance(search_speaker_a_results, list):
        search_speaker_a_results = search_speaker_a_results[:3]  # 再次限制为 3 条（双重保险）
        context = "\n".join([str(m) for m in search_speaker_a_results]) if search_speaker_a_results else "No relevant memories."
    else:
        context = str(search_speaker_a_results) if search_speaker_a_results else "No relevant memories."
    
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def search_query(client, query, metadata, frame, version, top_k=20, memory_version=None):
    speaker_a_user_id = metadata.get("speaker_a_user_id")

    if frame == "mem0":
        context, duration_ms = mem0_search(client, query, speaker_a_user_id, top_k)
    elif frame == "mem0_graph":
        context, duration_ms = mem0_graph_search(client, query, speaker_a_user_id, top_k)
    elif "memos-api" in frame:
        context, duration_ms = memos_api_search(client, query, speaker_a_user_id, top_k)
    elif frame == "memobase":
        context, duration_ms = memobase_search(client, query, speaker_a_user_id, top_k)
    elif frame == "memu":
        context, duration_ms = memu_search(client, query, speaker_a_user_id, top_k)
    elif frame == "supermemory":
        conv_idx = metadata["conv_idx"]
        mem_version = memory_version if memory_version is not None else version
        speaker_a_user_id = f"lcm{conv_idx}a_{mem_version}"
        context, duration_ms = supermemory_search(client, query, speaker_a_user_id, top_k)
    return context, duration_ms


def load_existing_results(frame, version, group_idx):
    result_path = (
        f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json"
    )
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f), True
        except Exception as e:
            print(f"Error loading existing results for group {group_idx}: {e}")
    return {}, False


def process_qa_file(conv_idx, qa_data, history_data, frame, version, top_k=20, num_workers=1, memory_version=None):
    """
        Args:
        conv_idx: conversation index
        qa_data: QA data
        history_data: history data
        frame: memory system framework
        version: version identifier
        top_k: top-k results
        num_workers: number of parallel workers
        memory_version: memory version identifier
    """
    search_results = defaultdict(list)
    conversation = history_data[conv_idx]["conversation"]
    speaker_a = conversation.get("speaker_a")
    mem_version = memory_version if memory_version is not None else version
    speaker_a_user_id = f"locomo_exp_user_{conv_idx}_speaker_a_{speaker_a}_{mem_version}"
    conv_id = f"locomo_exp_user_{conv_idx}"
    
    print(f"DEBUG: Processing QA file for conv_idx={conv_idx}, speaker_a={speaker_a}, user_id={speaker_a_user_id}", file=sys.stderr)
    print(f"DEBUG: Building user_id for search: {speaker_a_user_id} (speaker_a={speaker_a}, memory_version={mem_version}, version={version})", file=sys.stderr)

    existing_results, loaded = load_existing_results(frame, version, conv_idx)
    if loaded:
        print(f"Loaded existing results for group {conv_idx}")
        return existing_results

    client = None
    if frame == "mem0" or frame == "mem0_graph":
        from utils.client import Mem0Client

        client = Mem0Client(enable_graph="graph" in frame)
    elif frame == "memos-api":
        from utils.client import MemosApiClient

        client = MemosApiClient()
    elif frame == "memos-api-online":
        from utils.client import MemosApiOnlineClient

        client = MemosApiOnlineClient()
    elif frame == "memobase":
        from utils.client import MemobaseClient

        client = MemobaseClient()
    elif frame == "memu":
        from utils.client import MemuClient

        client = MemuClient()
    elif frame == "supermemory":
        from utils.client import SupermemoryClient

        client = SupermemoryClient()

    metadata = {
        "speaker_a": speaker_a,
        "speaker_a_user_id": speaker_a_user_id,
        "conv_idx": conv_idx,
        "conv_id": conv_id,
    }

    def process_qa(qa):
        if isinstance(qa, dict):
            query = qa.get("input") or qa.get("question")
        else:
            query = qa
            
        if not query:
            print(f"Warning: Empty query in QA data")
            return None
            
        context, duration_ms = search_query(client, query, metadata, frame, version, top_k=top_k, memory_version=memory_version)

        if not context:
            print(f"No context found for query: {query}")
            context = ""
        return {"query": query, "context": context, "duration_ms": duration_ms, "qa_data": qa}

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for qa in qa_data:
            futures.append(executor.submit(process_qa, qa))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Processing harmful QAs"
        ):
            result = future.result()
            if result:
                search_results[conv_id].append(result)

    os.makedirs(f"results/locomo/{frame}-{version}/tmp/", exist_ok=True)
    with open(
        f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{conv_idx}.json", "w"
    ) as f:
        json.dump(dict(search_results), f, indent=2)
        print(f"Save search results {conv_idx}")

    return search_results


def main(frame, version="default", num_workers=1, top_k=20, qa_data_path=None, history_data_path=None, memory_version=None):
    """
        Args:
        frame: memory system framework
        version: version identifier
        num_workers: number of parallel workers
        top_k: top-k results
        qa_data_path: QA data file path
        history_data_path: history data file path
        memory_version: memory version identifier
    """
    load_dotenv()
    
    if qa_data_path is None:
        raise ValueError("QA data path must be provided (--qa_data_path)")
    
    if history_data_path is None:
        raise ValueError("History data path must be provided (--history_data_path)")
    
    print(f"Loading QA data from: {qa_data_path}")
    with open(qa_data_path, 'r') as f:
        qa_data = json.load(f)
    
    print(f"Loading history data from: {history_data_path}")
    with open(history_data_path, 'r') as f:
        history_data = json.load(f)

    os.makedirs(f"results/locomo/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)

    conv_idx = 0
    print(f"Processing harmful QAs for user {conv_idx}...")
    print(f"Using memory_version: {memory_version if memory_version else version} for user_id construction")
    user_results = process_qa_file(conv_idx, qa_data, history_data, frame, version, top_k, num_workers, memory_version)
    for conv_id, results in user_results.items():
        all_search_results[conv_id].extend(results)

    with open(f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json", "w") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print("Save all search results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=[
            "mem0",
            "mem0_graph",
            "memos-api",
            "memos-api-online",
            "memobase",
            "memu",
            "supermemory",
        ],
        default="memos-api",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of parallel workers to process users"
    )
    parser.add_argument(
        "--top_k", type=int, default=15, help="Number of results to retrieve in search queries"
    )
    parser.add_argument(
        "--qa_data_path",
        type=str,
        required=True,
        help="Path to the QA data file (harmful questions)"
    )
    parser.add_argument(
        "--history_data_path",
        type=str,
        required=True,
        help="Path to the history conversation data file (LoCoMo format JSON)"
    )
    parser.add_argument(
        "--memory_version",
        type=str,
        default=None,
        help="Memory version identifier for user_id construction (defaults to version if not provided)"
    )
    args = parser.parse_args()
    lib = args.lib
    version = args.version
    workers = args.workers
    top_k = args.top_k
    qa_data_path = args.qa_data_path
    history_data_path = args.history_data_path
    memory_version = args.memory_version

    main(lib, version, workers, top_k, qa_data_path, history_data_path, memory_version)

