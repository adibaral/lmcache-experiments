import argparse
from tqdm.auto import tqdm
import json
import random
import time
from transformers import AutoTokenizer
from utils.client import VLLMClient

def load_queries(file_path: str) -> list[str]:
    """
    Load queries from a file, where each line is a separate query.
    
    Args:
        file_path (str): Path to the file containing queries.
    """
    with open(file_path, 'r') as file:
        queries = [line.strip() for line in file if line.strip()]
    return queries

def load_contexts(file_paths: list[str], truncate:bool=True, model_path: str=None, max_ctx_tokens: int = 10800, safety_margin:int=2048) -> list[str]:
    """
    Load contexts from a file.

    Args:
        file_paths (list[str]): List of paths to the context files.
        truncate (bool): Whether to truncate the context to fit within the model's context length.
        model_path (str): Model identifier to determine max context length.
        max_ctx_tokens (int): Maximum context tokens allowed, default is 4096.
        safety_margin (int): Safety margin to keep below the model's max context length.
    Returns:
        list[str]: List of contexts loaded from the files, truncated if specified.
    """
    raw_contexts = list()
    for file_path in file_paths:
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                context_obj = json.load(f)
            raw = json.dumps(context_obj, indent=4)
        else:
            with open(file_path, "r") as f:
                raw = f.read()
        raw_contexts.append(raw.strip())

    if not truncate:
        return raw_contexts

    num_contexts = len(raw_contexts)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model_max_length = tokenizer.model_max_length if tokenizer.model_max_length > 0 else max_ctx_tokens
    max_allowed_tokens = min(model_max_length - safety_margin, max_ctx_tokens)
    
    truncated_contexts = list()
    tokens_per_context = max_allowed_tokens // num_contexts
    for raw_context in raw_contexts:
        encoded_ids = tokenizer.encode(raw_context, add_special_tokens=False, truncation=True, max_length=tokens_per_context)
        truncated_context = tokenizer.decode(encoded_ids, skip_special_tokens=True)
        truncated_contexts.append(truncated_context.strip())
    return truncated_contexts


def build_history_from_contexts(contexts: list[str], system_prompt: str) -> list[dict]:
    """
    Build the chat history from the contexts and system prompt.
    
    Args:
        contexts (list[str]): List of context strings to include in the chat history.
        system_prompt (str): The system prompt to initialize the chat.
    
    Returns:
        list[dict]: The chat history as a list of dictionaries.
    """
    if contexts:
        random.shuffle(contexts)
        context = "\n\n".join(contexts)
        return [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"I've got a document:\n```\n{context}\n```"},
            {"role": "assistant", "content": "I've got your document."},
        ]
    else:
        return [{"role": "system", "content": system_prompt}]


def run_query_benchmark(model_path:str, client: VLLMClient, queries: list[str], contexts: list[str], system_prompt: str):
    """
    Run the benchmark per query by sending queries to the VLLM server and collecting results.

    Args:
        model_path (str): The path to the model to use for chat completions.
        client (VLLMClient): The client to interact with the VLLM server.
        queries (list[str]): List of queries to send to the server.
        contexts (list[str]): List of context strings to include in the chat history.
        system_prompt (str): System prompt to initialize the chat.
    """
    results = list()
    total_runs = len(queries) * 2  # Each query will be run twice (once with a cold start, once with a warm start)
    progress_bar = tqdm(range(total_runs))
    for i, query in enumerate(queries):
        query_results = {
            "_id": i,
            "query": query,
            "cold": dict(),
            "warm": dict(),
        }
        progress_bar.set_description(f"Processing query {i + 1}/{len(queries)}")
        
        # Create the chat history with the system prompt and context
        history = build_history_from_contexts(contexts, system_prompt)

        try:
            # Cold start: reset history and send query
            client.flush_kv_cache(model_path)
            client.set_history(history)
            cold_metrics = client.chat(query, model_path, temperature=0.0)
            query_results["cold"] = cold_metrics
            print(f"Cold start metrics for query {i+1}: {cold_metrics}")
            progress_bar.update(1)
            time.sleep(5)

            # Warm start: reuse history and send query
            client.set_history(history)
            warm_metrics = client.chat(query, model_path, temperature=0.0)
            query_results["warm"] = warm_metrics
            print(f"Warm start metrics for query {i+1}: {warm_metrics}")
            progress_bar.update(1)
            time.sleep(10)
            results.append(query_results)
        except Exception as e:
            print(f"Error processing query {i+1}: {e}")
            results.append({
                "_id": i,
                "query": query,
                "error": str(e),
            })
            progress_bar.update(2)
    
    progress_bar.close()
    return results    


def run_session_benchmark(model_path: str, client: VLLMClient, queries: list[str], contexts: list[str], system_prompt: str):
    """
    Run the benchmark for a session by sending all queries to the VLLM server and collecting results.
    Args:
        model_path (str): The path to the model to use for chat completions.
        client (VLLMClient): The client to interact with the VLLM server.
        queries (list[str]): List of queries to send to the server.
        contexts (list[str]): List of context strings to include in the chat history.
        system_prompt (str): System prompt to initialize the chat.
    """
    results = list()
    total_runs = len(queries)
    history = build_history_from_contexts(contexts, system_prompt)
    client.set_history(history)
    progress_bar = tqdm(range(total_runs), desc="Running cold queries")
    
    try:
        # Cold start: reset history and send all queries
        client.flush_kv_cache(model_path)
        start_time = time.perf_counter()
        for i, query in enumerate(queries):
            query_result = {
                "_id": i,
                "query": query,
            }
            metrics = client.chat(query, model_path, temperature=0.0)
            query_result["metrics"] = metrics
            results.append(query_result)
            progress_bar.update(1)
        end_time = time.perf_counter()
        duration = end_time - start_time
    except Exception as e:
        print(f"Error during cold queries: {e}")
        results.append({
            "error": str(e),
        })

    progress_bar.close()
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script for LMCache with Redis backend running on a vLLM server.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the model to use for chat completions",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost",
        help="Host URL of the vLLM server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number of the VLLM server",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt to initialize the chat",
    )
    parser.add_argument(
        "--context-files",
        nargs="+",
        default=list(),
        help="List of paths to context files to use for the chat.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to the prompt file to use for the chat.",
    )
    parser.add_argument(
        "--mode",
        choices=["query", "session"],
        help="Benchmark mode: 'query' measures cold vs warm for each query. 'session' measures total time for all queries with and without cache."
    )
    parser.add_argument(
        "--max-ctx-tokens",
        type=int,
        default=10800,
        help="Maximum context tokens allowed, default is 10800.",
    )
    parser.add_argument(
        "--safety-margin",
        type=int,
        default=2048,
        help="Safety margin to leave free at the end of the context, default is 2048.",
    )
    parser.add_argument(
        "--truncate-context",
        action="store_true",
        help="Whether to truncate the context to fit within the model's context length.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark_results.json",
        help="Path to the output file where benchmark results will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    print(args)

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load queries and context if provided
    queries = load_queries(args.prompt_file)
    contexts = load_contexts(
        args.context_files,
        truncate=args.truncate_context,
        model_path=args.model_path,
        max_ctx_tokens=args.max_ctx_tokens,
        safety_margin=args.safety_margin,
    )
    
    # Initialize the VLLM client
    client = VLLMClient(host=args.host, port=args.port, system_prompt=args.system_prompt)
    
    # Run the benchmark
    if args.mode == "query":
        results = run_query_benchmark(
            model_path=args.model_path,
            client=client,
            queries=queries,
            contexts=contexts,
            system_prompt=args.system_prompt,
        )
    elif args.mode == "session":
        results = run_session_benchmark(
            model_path=args.model_path,
            client=client,
            queries=queries,
            contexts=contexts,
            system_prompt=args.system_prompt,
        )
    
    # Save results to output file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
