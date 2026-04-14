import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

import nltk
import numpy as np
import tiktoken
import transformers

from bert_score import score as bert_score
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def setup_logging(log_file=None):
    if log_file is None:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"locomo_eval_{timestamp}.log")
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


logger, log_file = setup_logging()

logging.basicConfig(level=logging.CRITICAL)
transformers.logging.set_verbosity_error()
encoding = tiktoken.get_encoding("cl100k_base")
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Warning: Failed to download NLTK resources: {e}")

try:
    sentence_model_name = "Qwen/Qwen3-Embedding-0.6B"
    sentence_model = SentenceTransformer(sentence_model_name)
    print(f"SentenceTransformer model : {sentence_model_name} loaded successfully.")
except Exception as e:
    print(f"Failed to load SentenceTransformer model: {e}")
    sentence_model = None


class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


async def locomo_grader(llm_client, question: str, gold_answer: str, response: str) -> bool:
    system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

    accuracy_prompt = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": accuracy_prompt},
                ],
                temperature=0,
            ),
            timeout=60.0
        )
        message_content = response.choices[0].message.content
        label = json.loads(message_content)["label"]
        parsed = LLMGrade(llm_judgment=label, llm_reasoning="")

        return parsed.llm_judgment.strip().lower() == "correct"
    except asyncio.TimeoutError:
        logger.warning(f"⏱️  API 超时（60s）: {question[:50]}...")
        return False
    except json.JSONDecodeError as e:
        logger.warning(f"❌ JSON 解析失败: {question[:50]}... Error: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  评分错误 '{question[:50]}...': {type(e).__name__}: {str(e)[:100]}")
        return False


def calculate_rouge_scores(gold_answer, response):
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, response)
        metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2_f"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Failed to calculate ROUGE scores: {e}")
    return metrics


def calculate_bleu_scores(gold_tokens, response_tokens):
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    try:
        smoothing = SmoothingFunction().method1
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except ZeroDivisionError:
        pass
    except Exception as e:
        print(f"Failed to calculate BLEU scores: {e}")

    return metrics


def calculate_meteor_score(gold_tokens, response_tokens):
    try:
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        print(f"Failed to calculate METEOR score: {e}")
        return 0.0


def calculate_semantic_similarity(gold_answer, response):
    global sentence_model

    try:
        if sentence_model is None:
            sentence_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        gold_embedding = sentence_model.encode([gold_answer], show_progress_bar=False)[0]
        response_embedding = sentence_model.encode([response], show_progress_bar=False)[0]
        return 1 - cosine(gold_embedding, response_embedding)
    except Exception as e:
        print(f"Failed to calculate semantic similarity: {e}")
        return 0.0


def calculate_f1_score(gold_tokens, response_tokens):
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)

        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0

        precision = len(gold_set.intersection(response_set)) / len(response_set)
        recall = len(gold_set.intersection(response_set)) / len(gold_set)

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


def calculate_nlp_metrics(gold_answer, response, context, options=None):
    if options is None:
        options = ["lexical", "semantic"]

    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""

    metrics = {"context_tokens": len(encoding.encode(context)) if context else 0}

    if "lexical" in options:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())

        metrics["lexical"] = {}
        metrics["lexical"]["f1"] = calculate_f1_score(gold_tokens, response_tokens)
        metrics["lexical"].update(calculate_rouge_scores(gold_answer, response))
        metrics["lexical"].update(calculate_bleu_scores(gold_tokens, response_tokens))
        metrics["lexical"]["meteor"] = calculate_meteor_score(gold_tokens, response_tokens)

    if "semantic" in options:
        metrics["semantic"] = {}
        metrics["semantic"]["similarity"] = calculate_semantic_similarity(gold_answer, response)
        _, _, f1 = bert_score(
            [gold_answer], [response], lang="en", rescale_with_baseline=True, verbose=False
        )
        metrics["semantic"]["bert_f1"] = f1.item() if f1 is not None else 0.0

    return metrics


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


async def process_group_responses(group_id, group_responses, oai_client, options, num_runs: int):
    graded_responses = []
    total_questions = len(group_responses)
    
    logger.info(f"Start processing {group_id}, {total_questions} questions")
    start_time = time.time()

    # Process responses with asyncio for concurrent API calls
    for idx, response in enumerate(tqdm(group_responses, desc=f"Processing group {group_id}")):
        question = response.get("question")
        answer = response.get("answer")
        ground_truth = response.get("golden_answer")
        category = response.get("category")

        context = response.get("search_context", "")
        response_duration_ms = response.get("response_duration_ms", 0.0)
        search_duration_ms = response.get("search_duration_ms", 0.0)

        if ground_truth is None:
            continue

        if idx % 50 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (total_questions - idx) / rate if rate > 0 else 0
            logger.info(f"Progress: {idx}/{total_questions} ({100*idx/total_questions:.1f}%) | Elapsed time: {elapsed:.0f}s | Remaining time: {remaining:.0f}s")

        grading_tasks = [
            locomo_grader(oai_client, question, ground_truth, answer) for _ in range(num_runs)
        ]
        judgments = await asyncio.gather(*grading_tasks, return_exceptions=True)
        judgments = [j if not isinstance(j, Exception) else False for j in judgments]
        judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

        nlp_metrics = calculate_nlp_metrics(ground_truth, answer, context, options)

        graded_response = {
            "question": question,
            "answer": answer,
            "golden_answer": ground_truth,
            "category": category,
            "llm_judgments": judgments_dict,
            "nlp_metrics": nlp_metrics,
            "response_duration_ms": response_duration_ms,
            "search_duration_ms": search_duration_ms,
            "total_duration_ms": response_duration_ms + search_duration_ms,
        }
        graded_responses.append(graded_response)

    elapsed = time.time() - start_time
    logger.info(f"{group_id} processing completed: {len(graded_responses)} questions, time: {elapsed:.1f}s")
    return group_id, graded_responses


async def process_single_group(group_id, group_responses, oai_client, options, num_runs):
    try:
        start_time = time.time()
        result = await process_group_responses(
            group_id, group_responses, oai_client, options, num_runs
        )
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        logger.info(f"{group_id} processing completed, time: {elapsed_time}s")
        return result
    except Exception as e:
        logger.error(f"{group_id} processing error: {type(e).__name__}: {str(e)}")
        return group_id, []


async def main(frame, version="default", options=None, num_runs=1, max_workers=4):
    logger.info("=" * 80)
    logger.info(f"LoCoMo evaluation started")
    logger.info(f"   Framework: {frame}")
    logger.info(f"   Version: {version}")
    logger.info(f"   Number of times to run each question: {num_runs}")
    logger.info(f"   Maximum number of concurrent workers: {max_workers}")
    logger.info(f"   Log file: {log_file}")
    logger.info("=" * 80)

    results_dir = f"results/locomo/{frame}-{version}"
    response_path = f"{results_dir}/{frame}_locomo_responses.json"
    judged_path = f"{results_dir}/{frame}_locomo_judged.json"

    os.makedirs(results_dir, exist_ok=True)

    load_dotenv()
    logger.info("Loading OpenAI API configuration...")
    oai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )
    logger.info(f"API configuration completed: {os.getenv('OPENAI_BASE_URL')}")

    logger.info(f"Loading response file: {response_path}")
    with open(response_path) as file:
        locomo_responses = json.load(file)

    num_users = 10
    all_grades = {}

    total_responses_count = sum(
        len(locomo_responses.get(f"locomo_exp_user_{i}", [])) for i in range(num_users)
    )
    logger.info(f"Loaded {total_responses_count} responses from {num_users} users")

    logger.info("Preparing evaluation tasks...")
    tasks = []
    active_users = 0
    for group_idx in range(num_users):
        group_id = f"locomo_exp_user_{group_idx}"
        group_responses = locomo_responses.get(group_id, [])
        if not group_responses:
            logger.debug(f"{group_id}: No responses")
            continue

        active_users += 1
        tasks.append(process_single_group(group_id, group_responses, oai_client, options, num_runs))
        logger.info(f"{group_id}: {len(group_responses)} questions")

    logger.info(f"Start evaluating {active_users} user groups, using {max_workers} concurrent workers")
    logger.info(f"Total API calls: {total_responses_count * num_runs}")
    eval_start_time = time.time()

    semaphore = asyncio.Semaphore(max_workers)

    async def limited_task(task):
        async with semaphore:
            return await task

    limited_tasks = [limited_task(task) for task in tasks]
    group_results = await asyncio.gather(*limited_tasks)

    for group_id, graded_responses in group_results:
        all_grades[group_id] = graded_responses

    eval_elapsed = time.time() - eval_start_time
    logger.info(f"Evaluation completed! Time: {eval_elapsed:.1f}s ({eval_elapsed/60:.1f} minutes)")
    logger.info("Calculating final scores...")

    run_scores = []
    evaluated_count = 0
    if num_runs > 0:
        for i in range(1, num_runs + 1):
            judgment_key = f"judgment_{i}"
            current_run_correct_count = 0
            current_run_total_count = 0
            for group in all_grades.values():
                for response in group:
                    if judgment_key in response["llm_judgments"]:
                        if response["llm_judgments"][judgment_key]:
                            current_run_correct_count += 1
                        current_run_total_count += 1

            if current_run_total_count > 0:
                run_accuracy = current_run_correct_count / current_run_total_count
                run_scores.append(run_accuracy)
                logger.info(f"Run {i}: Accuracy {run_accuracy:.4f} ({current_run_correct_count}/{current_run_total_count})")

        evaluated_count = current_run_total_count

    logger.info("")
    if evaluated_count > 0:
        mean_of_scores = np.mean(run_scores)
        std_of_scores = np.std(run_scores)
        logger.info(f"Final results:")
        logger.info(f"   LLM-as-a-Judge average score: {mean_of_scores:.4f}")
        logger.info(f"   Standard deviation: {std_of_scores:.4f}")
        logger.info(f"   Based on {num_runs} runs, {evaluated_count} questions")
        logger.info(f"   Each run score: {[round(s, 4) for s in run_scores]}")
    else:
        logger.warning("No responses evaluated")
        logger.warning("LLM-as-a-Judge score: N/A (0/0)")

    all_grades = convert_numpy_types(all_grades)
    with open(judged_path, "w") as f:
        json.dump(all_grades, f, indent=2)
    
    file_size = os.path.getsize(judged_path) / (1024*1024)
    logger.info(f"Evaluation results saved")
    logger.info(f"File: {judged_path}")
    logger.info(f"Size: {file_size:.1f} MB")
    logger.info("=" * 80)
    logger.info("LoCoMo evaluation completed successfully!")
    logger.info("=" * 80)


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
        "--num_runs",
        type=int,
        default=3,
        help="Number of times to run the LLM grader for each question",
    )
    parser.add_argument("--options", nargs="+", default=["lexical"])
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of concurrent workers for processing groups"
    )
    args = parser.parse_args()

    asyncio.run(main(args.lib, args.version, args.options, args.num_runs, args.workers))
