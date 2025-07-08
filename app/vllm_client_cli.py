import time
from typing import Optional, Any
from openai import OpenAI
import requests
import argparse

# from util.log_metrics_collector import LogMetricsCollector


class VLLMClient:
    def __init__(
        self,
        host: str = "http://localhost",
        port: int = 8000,
        system_prompt: str = "You are a helpful assistant.",
        log_path: str = "vllm.log",
    ):
        self.host = host
        self.port = port
        self.api_base_url = f"{host}:{port}"
        self.system_prompt = system_prompt
        self.log_path = log_path
        self.history = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"{self.api_base_url}/v1/",
        )
        # self.log_metrics_collector = LogMetricsCollector(self.log_path)

    def chat(
        self,
        prompt: str,
        model_path: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Send a chat message to the model and return the response.
        Args:
            prompt (str): The input message to send to the model.
            model_path (str): The specific model path to use for the request.
            temperature (Optional[float]): Sampling temperature to use for the response generation.
                If None, the default temperature set in the client will be used.
            max_tokens (Optional[int]): Maximum number of tokens to generate in the response.
                If None, the default max tokens set in the client will be used.

        Returns:
            dict: A dictionary containing the model's response and token usage information.
        """
        self.history.append({"role": "user", "content": prompt})
        start_time = time.perf_counter()
        request_id, response_time, tfft, content, total_tokens, completion_tokens, prompt_tokens = None, None, None, None, None, None, None
        
        if stream:
            response = self.client.chat.completions.create(
                model=model_path or self.model_path,
                messages=self.history,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            chunk_messages = list()
            print("llm > ", end="", flush=True)
            completion_tokens = 0
            content = ""
            for chunk in response:
                chunk_message = chunk.choices[0].delta.content
                if chunk_message:
                    completion_tokens += 1
                    if tfft is None:
                        tfft = time.perf_counter()
                    print(chunk_message, end="", flush=True)
                    chunk_messages.append(chunk_message)
                    content += chunk_message
            tfft = tfft - start_time
            response_time = time.perf_counter() - start_time
            print()
            
            tokens_response = requests.post(
                f"{self.api_base_url}/tokenize",
                json={"prompt": prompt}
            )
            prompt_tokens = tokens_response.json().get("count", 0)
            request_id = chunk.id
            total_tokens = prompt_tokens + completion_tokens

        else:
            response = self.client.chat.completions.create(
                model=model_path or self.model_path,
                messages=self.history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_time = time.perf_counter() - start_time
            request_id = response.id
            content = response.choices[0].message.content
            total_tokens = response.usage.total_tokens
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
        
        self.history.append(
            {"role": "assistant", "content": content}
        )
        # TODO: Find a way to extract cache hit and other metrics
        # log_metrics = self.log_metrics_collector.metrics_by_rid.pop(request_id, {})
        # throughput = self.log_metrics_collector.throughput_stats
        
        return {
            "request_id": request_id,
            "prompt": prompt,
            "content": content,
            "total_tokens": total_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tfft": tfft,
            "response_time": response_time,
            # **log_metrics,
            # **throughput,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM Client CLI")
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost",
        help="Host URL of the VLLM server",
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
        "--log-path",
        type=str,
        default="vllm.log",
        help="Path to the log file for metrics collection",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the model to use for chat completions",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode for chat completions",
    )
    args = parser.parse_args()

    llm = VLLMClient(
        host=args.host,
        port=args.port,
        system_prompt=args.system_prompt,
    )

    while True:
        _input = input(">>> ")
        if _input.lower() in ["exit", "quit", "q"]:
            break
        response = llm.chat(_input, model_path=args.model_path, stream=args.stream, temperature=0.0)
        print(response)
        print("-" * 80)
