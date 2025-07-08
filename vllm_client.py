import time
from typing import Optional, Any
from openai import OpenAI

from log_metrics_collector import LogMetricsCollector


class VLLMClient:
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        host: str = "http://localhost",
        port: int = 8000,
        system_prompt: str = "You are a helpful assistant.",
        log_path: str = "vllm.log",
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.api_base_url = f"{self.host}:{self.port}/v1/"
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
            base_url=self.api_base_url,
        )
        self.log_metrics_collector = LogMetricsCollector(self.log_path)

    def chat(
        self,
        prompt: str,
        model_path: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Send a chat message to the model and return the response.
        Args:
            prompt (str): The input message to send to the model.
            model_path (Optional[str]): The specific model path to use for the request.
                If None, the default model path set during initialization will be used.
            temperature (Optional[float]): Sampling temperature to use for the response generation.
                If None, the default temperature set in the client will be used.
            max_tokens (Optional[int]): Maximum number of tokens to generate in the response.
                If None, the default max tokens set in the client will be used.

        Returns:
            dict: A dictionary containing the model's response and token usage information.
        """
        self.history.append({"role": "user", "content": prompt})
        start_time = time.perf_counter()
        response_time, tfft = None, None

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
            for chunk in response:
                chunk_message = chunk.choices[0].delta.content
                if chunk_message:
                    if tfft is None:
                        tfft = time.perf_counter()
                    print(chunk_message, end="", flush=True)
                    chunk_messages.append(chunk_message)
            response_time = time.perf_counter() - start_time
            print()
            self.history.append(
                {"role": "assistant", "content": "".join(chunk_messages)}
            )

        else:
            response = self.client.chat.completions.create(
                model=model_path or self.model_path,
                messages=self.history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_time = time.perf_counter() - start_time
            self.history.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
        
        request_id = response.id
        log_metrics = self.log_metrics_collector.metrics_by_rid.pop(request_id, {})
        throughput = self.log_metrics_collector.throughput_stats
        return {
            "request_id": request_id,
            "prompt": prompt,
            "content": response.choices[0].message.content,
            "total_tokens": response.usage.total_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "tfft": tfft,
            "response_time": response_time,
            **log_metrics,
            **throughput,
        }


if __name__ == "__main__":
    llm = VLLMClient(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        host="http://localhost",
        port=8000,
        system_prompt="You are a helpful assistant.",
    )

    while True:
        _input = input(">>> ")
        if _input.lower() in ["exit", "quit", "q"]:
            break
        response = llm.chat(_input, stream=True)
        print(response)
