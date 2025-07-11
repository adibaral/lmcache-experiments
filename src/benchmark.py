import argparse
from tqdm.auto import tqdm
import json
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

def load_context(file_path: str, truncate:bool=True, model_path: str=None, max_ctx_tokens: int = 10800, safety_margin:int=2048) -> str:
    """
    Load context from a file.

    Args:
        file_path (str): Path to the context file.
        truncate (bool): Whether to truncate the context to fit within the model's context length.
        model_path (str): Model identifier to determine max context length.
        max_ctx_tokens (int): Maximum context tokens allowed, default is 4096.
        safety_margin (int): Safety margin to keep below the model's max context length.
    Returns:
        str: Truncated context string.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            context_obj = json.load(f)
        raw_context = json.dumps(context_obj, indent=4)
    else:
        with open(file_path, 'r') as f:
            raw_context = f.read()
    
    if not truncate:
        return raw_context.strip()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model_max_length = tokenizer.model_max_length if tokenizer.model_max_length > 0 else max_ctx_tokens
        max_allowed_tokens = max_ctx_tokens - safety_margin
        max_allowed_tokens = min(model_max_length - safety_margin, max_allowed_tokens)
        encoded_ids = tokenizer.encode(raw_context, add_special_tokens=False, truncation=True, max_length=max_allowed_tokens)
        truncated_context = tokenizer.decode(encoded_ids, skip_special_tokens=True)
    except Exception:
        char_limit = (max_ctx_tokens - safety_margin) * 4
        truncated_context = raw_context[:char_limit]

    return truncated_context.strip()


def build_history_from_context(context: str, system_prompt: str) -> list[dict]:
    """
    Build the chat history from the context and system prompt.
    
    Args:
        context (str): The context to include in the chat history.
        system_prompt (str): The system prompt to initialize the chat.
    
    Returns:
        list[dict]: The chat history as a list of dictionaries.
    """
    if context:
        return [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"I've got a document:\n```\n{context}\n```"},
            {"role": "assistant", "content": "I've got your document."},
        ]
    else:
        return [{"role": "system", "content": system_prompt}]


def run_query_benchmark(model_path:str, client: VLLMClient, queries: list[str], context: str, system_prompt: str):
    """
    Run the benchmark per query by sending queries to the VLLM server and collecting results.

    Args:
        model_path (str): The path to the model to use for chat completions.
        client (VLLMClient): The client to interact with the VLLM server.
        queries (list[str]): List of queries to send to the server.
        context (str): Context to set in the chat history.
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
        history = build_history_from_context(context, system_prompt)

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
        time.sleep(5)

        results.append(query_results)
    
    progress_bar.close()
    return results    


def run_session_benchmark(model_path: str, client: VLLMClient, queries: list[str], context: str, system_prompt: str):
    """
    Run the benchmark for a session by sending all queries to the VLLM server and collecting results.
    Args:
        model_path (str): The path to the model to use for chat completions.
        client (VLLMClient): The client to interact with the VLLM server.
        queries (list[str]): List of queries to send to the server.
        context (str): Context to set in the chat history.
        system_prompt (str): System prompt to initialize the chat.
    """
    results = list()
    total_runs = len(queries)
    
    # Cold start: reset history and send all queries
    client.flush_kv_cache(model_path)
    history = build_history_from_context(context, system_prompt)
    client.set_history(history)
    progress_bar = tqdm(range(total_runs), desc="Running cold queries")
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
    results.append({
        "duration": duration,
        "total_queries": len(queries),
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
    # TODO: What is max_ctx_tokens in their benchmark?
    parser.add_argument(
        "--context-file",
        type=str,
        default=None,
        help="Path to the context file to use for the chat. If not provided, no context will be used.",
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
    args = parser.parse_args()

    # Load queries and context if provided
    queries = load_queries(args.prompt_file)
    context = load_context(
        args.context_file,
        truncate=args.truncate_context,
        model_path=args.model_path,
        max_ctx_tokens=args.max_ctx_tokens,
        safety_margin=args.safety_margin,
    ) if args.context_file else ""
    
    # Initialize the VLLM client
    client = VLLMClient(host=args.host, port=args.port, system_prompt=args.system_prompt)
    
    # Run the benchmark
    if args.mode == "query":
        results = run_query_benchmark(
            model_path=args.model_path,
            client=client,
            queries=queries,
            context=context,
            system_prompt=args.system_prompt,
        )
    elif args.mode == "session":
        results = run_session_benchmark(
            model_path=args.model_path,
            client=client,
            queries=queries,
            context=context,
            system_prompt=args.system_prompt,
        )
    
    # Save results to output file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
