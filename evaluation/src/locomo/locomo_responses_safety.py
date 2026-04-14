import argparse
import asyncio
import json
import os
import sys

from time import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import APITimeoutError, APIError
from tqdm import tqdm


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
EVAL_SCRIPTS_DIR = os.path.join(ROOT_DIR, "evaluation", "scripts")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, EVAL_SCRIPTS_DIR)


async def locomo_response(frame, llm_client, context, question: str, model: str, log_prompt=True, max_retries=3, timeout=120) -> tuple:
    """
    Args:
        context: can be a string (old logic) or a dictionary (new logic, containing memory and personality)
    """
    
    user_traits = "No user traits."
    memory_text = ""
    
    if isinstance(context, dict):
        memory_text = context.get("memory", "No relevant memories.")
        user_traits = context.get("personality", "No user traits.")
    else:
        memory_text = str(context) if context else "No relevant memories."
    
    user_prompt = f"""Reply in a natural, spoken tone, optionally using relevant memory or user personality details when appropriate.\n""" \
        + f"""Memory:\n{memory_text}\n""" \
        + f"""User's personality:\n{user_traits}\n""" \
        + f"""User's latest input: {question}\n"""
    
    if log_prompt:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f" QA Prompt Log", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"Model: {model}", file=sys.stderr)
        print(f"Question: {question}", file=sys.stderr)
        print(f"Context (Memory):\n{context}", file=sys.stderr)
        print(f"User's personality: No user traits.", file=sys.stderr)
        print(f"\n--- Full Prompt (user_prompt) ---", file=sys.stderr)
        print(user_prompt, file=sys.stderr)
        print(f"--- End of Prompt ---", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    # extra_body={
                    #     "enable_thinking": False
                    # }
                ),
                timeout=timeout
            )
            result = response.choices[0].message.content or ""
            return result, user_prompt
        except asyncio.TimeoutError:
            last_error = f"API调用超时 (timeout={timeout}s)"
            print(f"⚠️  警告: {last_error} - 问题: {question[:50]}... (尝试 {attempt + 1}/{max_retries})", file=sys.stderr)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except APITimeoutError as e:
            last_error = f"API超时错误: {str(e)}"
            print(f"⚠️  警告: {last_error} - 问题: {question[:50]}... (尝试 {attempt + 1}/{max_retries})", file=sys.stderr)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except APIError as e:
            last_error = f"API错误: {str(e)}"
            print(f"⚠️  警告: {last_error} - 问题: {question[:50]}... (尝试 {attempt + 1}/{max_retries})", file=sys.stderr)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            last_error = f"未知错误: {str(e)}"
            print(f"❌ 错误: {last_error} - 问题: {question[:50]}... (尝试 {attempt + 1}/{max_retries})", file=sys.stderr)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    
    print(f"❌ 失败: 所有重试均失败 - {last_error} - 问题: {question[:50]}...", file=sys.stderr)
    return f"[ERROR: Failed to generate response after {max_retries} attempts. {last_error}]", user_prompt


async def process_qa(frame, qa_data, search_result, oai_client, model="gpt-4o-mini", timeout=120):
    """
        Args:
        frame: memory system framework
        qa_data: QA data (may contain input/question fields)
        search_result: search result
        oai_client: OpenAI client
        model: model name
        timeout: API call timeout time (seconds)
    """
    start = time()
    
    # 支持多种格式
    if isinstance(qa_data, dict):
        query = qa_data.get("input") or qa_data.get("question")
        source = qa_data.get("source", "unknown")
    else:
        query = qa_data
        source = "unknown"

    context = search_result.get("context")

    answer, user_prompt = await locomo_response(frame, oai_client, context, query, model, log_prompt=True, timeout=timeout)

    response_duration_ms = (time() - start) * 1000

    print(f"Processed question: {query[:50]}...", file=sys.stderr)
    print(f"Answer: {answer[:100]}...", file=sys.stderr)
    
    return {
        "input": query,
        "output": answer,
        "source": source,
        "search_context": search_result.get("context", ""),
        "response_duration_ms": response_duration_ms,
        "search_duration_ms": search_result.get("duration_ms", 0),
        "user_prompt": user_prompt,
    }


async def main(frame, version="default", num_workers=1, model=None, api_key=None, base_url=None, timeout=120, batch_size=10):
    """
        Args:
        frame: memory system framework
        version: version identifier
        num_workers: number of parallel workers (currently not used, for compatibility)
        model: model name (if None, will use CHAT_MODEL from env)
        api_key: API key (if None, will use CHAT_MODEL_API_KEY from env)
        base_url: API base URL (if None, will use CHAT_MODEL_BASE_URL from env)
        timeout: API call timeout time (seconds)
        batch_size: batch size, avoid too many concurrent requests at once
    """
    search_path = f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json"
    
    load_dotenv()
    
    final_model = model if model else os.getenv("CHAT_MODEL", "gpt-4o-mini")
    final_api_key = api_key if api_key else os.getenv("CHAT_MODEL_API_KEY")
    final_base_url = base_url if base_url else os.getenv("CHAT_MODEL_BASE_URL")
    
    model_safe = final_model.replace("/", "_").replace(":", "_")
    response_path = f"results/locomo/{frame}-{version}/{frame}_locomo_responses_{model_safe}.json"
    
    oai_client = AsyncOpenAI(
        api_key=final_api_key, 
        base_url=final_base_url,
        timeout=timeout
    )
    
    print(f"Using model: {final_model}")
    print(f"API base URL: {final_base_url}")
    print(f"Timeout: {timeout}s per request")
    print(f"Batch size: {batch_size}")

    with open(search_path) as file:
        locomo_search_results = json.load(file)

    all_responses = {}
    
    group_idx = 0
    group_id = f"locomo_exp_user_{group_idx}"
    search_results = locomo_search_results.get(group_id, [])
    
    if not search_results:
        print(f"Warning: No search results found for {group_id}")
        return

    responses = []
    total = len(search_results)
    
    print(f"Processing {total} questions in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = search_results[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} (question {i+1}-{min(i+batch_size, total)})", file=sys.stderr)
        
        tasks = [
            process_qa(frame, result.get("qa_data"), result, oai_client, model=final_model, timeout=timeout)
            for result in batch
        ]
        
        try:
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for idx, response in enumerate(batch_responses):
                if isinstance(response, Exception):
                    print(f"Batch {batch_num} task {idx+1} failed: {response}", file=sys.stderr)
                    qa_data = batch[idx].get("qa_data", {})
                    if isinstance(qa_data, dict):
                        query = qa_data.get("input") or qa_data.get("question", "unknown")
                    else:
                        query = str(qa_data)
                    batch_responses[idx] = {
                        "input": query,
                        "output": f"[ERROR: {str(response)}]",
                        "source": qa_data.get("source", "unknown") if isinstance(qa_data, dict) else "unknown",
                        "search_context": batch[idx].get("context", ""),
                        "response_duration_ms": 0,
                        "search_duration_ms": batch[idx].get("duration_ms", 0),
                        "user_prompt": "",
                    }
            
            responses.extend(batch_responses)
            print(f"Batch {batch_num}/{total_batches} completed", file=sys.stderr)
            
        except Exception as e:
            print(f"Batch {batch_num} processing failed: {e}", file=sys.stderr)
            for result in batch:
                qa_data = result.get("qa_data", {})
                if isinstance(qa_data, dict):
                    query = qa_data.get("input") or qa_data.get("question", "unknown")
                else:
                    query = str(qa_data)
                responses.append({
                    "input": query,
                    "output": f"[ERROR: Batch processing failed: {str(e)}]",
                    "source": qa_data.get("source", "unknown") if isinstance(qa_data, dict) else "unknown",
                    "search_context": result.get("context", ""),
                    "response_duration_ms": 0,
                    "search_duration_ms": result.get("duration_ms", 0),
                    "user_prompt": "",
                })
    
    all_responses[group_id] = responses

    os.makedirs(os.path.dirname(response_path), exist_ok=True)

    with open(response_path, "w") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
        print(f"Saved response results to {response_path}")
        print(f"Total responses generated: {len(responses)}")
        
        error_count = sum(1 for r in responses if isinstance(r, dict) and r.get("output", "").startswith("[ERROR:"))
        if error_count > 0:
            print(f"Warning: {error_count} responses generated failed")


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
        help="Version identifier for loading results (e.g., 1010)",
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of parallel workers (kept for compatibility)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for generating responses (if not provided, will use CHAT_MODEL from env)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (if not provided, will use CHAT_MODEL_API_KEY from env)"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="API base URL (if not provided, will use CHAT_MODEL_BASE_URL from env)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="API call timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for concurrent requests (default: 10)"
    )
    args = parser.parse_args()
    lib = args.lib
    version = args.version
    workers = args.workers
    model = args.model
    api_key = args.api_key
    base_url = args.base_url
    timeout = args.timeout
    batch_size = args.batch_size
    asyncio.run(main(lib, version, workers, model, api_key, base_url, timeout, batch_size))

