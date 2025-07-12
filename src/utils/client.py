import time
from typing import Any
from openai import OpenAI
import requests
from tqdm.auto import tqdm
import random
import string

class VLLMClient:
    def __init__(
        self,
        host: str = "http://localhost",
        port: int = 8000,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.host = host
        self.port = port
        self.api_base_url = f"{host}:{port}"
        self.system_prompt = system_prompt
        self.history = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"{self.api_base_url}/v1/",
            max_retries=10,
            timeout=240,
        )

    def chat(
        self,
        prompt: str,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stream: bool = True,
    ) -> dict[str, Any]:
        """
        Send a chat message to the model and return the response.
        Args:
            prompt (str): The input message to send to the model.
            model_path (str): The specific model path to use for the request.
            temperature (float): Sampling temperature to use for the response generation.
            max_tokens (int): Maximum number of tokens to generate in the response.
            stream (bool): Whether to stream the response or not.

        Returns:
            dict: A dictionary containing the model's response and token usage information.
        """
        self.history.append({"role": "user", "content": prompt})
        request_id, response_time, tfft, content, total_tokens, completion_tokens, prompt_tokens = None, None, None, None, None, None, None
        
        if stream:
            start_time = time.perf_counter()
            response = self.client.chat.completions.create(
                model=model_path,
                messages=self.history,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            completion_tokens = 0
            content = ""
            for chunk in response:
                chunk_message = chunk.choices[0].delta.content
                if chunk_message:
                    completion_tokens += 1
                    if tfft is None:
                        tfft = time.perf_counter()
                    content += chunk_message
            tfft = tfft - start_time
            response_time = time.perf_counter() - start_time
            
            tokens_response = requests.post(
                f"{self.api_base_url}/tokenize",
                json={"prompt": prompt}
            )
            prompt_tokens = tokens_response.json().get("count", 0)
            request_id = chunk.id
            total_tokens = prompt_tokens + completion_tokens

        else:
            start_time = time.perf_counter()
            response = self.client.chat.completions.create(
                model=model_path,
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
        
        return {
            "request_id": request_id,
            "prompt": prompt,
            "content": content,
            "total_tokens": total_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tfft": tfft,
            "response_time": response_time,
        }

    def reset_history(self, system:bool=True) -> None:
        """
        Reset the chat history to the initial state. If `system` is True, it will reset to the system prompt.
        Args:
            system (bool): If True, the history will include the system prompt. If False, it will be empty.
        """
        if system:
            self.history = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            ]
        else:
            self.history = []

    def set_history(self, history: list[dict[str, str]]) -> None:
        """Set the chat history to a specific value.
        Args:
            history (list[dict[str, str]]): The chat history to set.
        """
        self.history = history

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set a new system prompt for the chat history.
        Args:
            system_prompt (str): The new system prompt to set.
        """
        self.system_prompt = system_prompt

    def flush_kv_cache(self, model_path: str, num_fillers: int = 20, filler_len_chars: int = 100_000):
        """
        Evict KV cache by sending large filler prompts.

        Args:
            model_path (str): The model to use.
            num_fillers (int): Number of filler prompts to send.
            filler_len_chars (int): Number of characters per filler prompt.
        """
        def rand_ascii(n: int) -> str:
            return "".join(random.choices(string.ascii_letters + string.digits, k=n))
        
        filler_text = rand_ascii(filler_len_chars)
        filler_messages = [
            {"role": "user", "content": f"I've got a document:\n```\n{filler_text}\n```"},
            {"role": "assistant", "content": "I've got your document."},
            {"role": "user", "content": "noop"},
        ]

        for _ in tqdm(range(num_fillers), desc="Evicting KV cache"):
            self.client.chat.completions.create(
                model=model_path,
                messages=filler_messages,
                temperature=0.0,
                max_tokens=1,
                stream=False,
            )
    