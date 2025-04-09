"""
Utility functions for the controlling LLM models.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
import time
from functools import reduce

import httpx


class GeminiBackend:
    def __init__(self, api_key: str, http_req_params: dict = {}, **kwargs) -> None:
        """
        Parameters
        ----------
        api_key
            The API key for the Gemini API.
        http_req_params
            Additional HTTP request parameters.
            Common parameters include `timeout`, `proxies`, etc.
        **kwargs
            Additional parameters.
        """
        super().__init__()
        self.api_key = api_key
        self.http_req_params = http_req_params
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.logger = logging.getLogger("phylox")
        self.num_retry = 5

    def restful_request(self, query: str):
        content_type = "application/json; charset=utf-8"
        body = {
            "contents": {"role": "user", "parts": {"text": query}},
        }
        try:
            response = httpx.post(
                self.url,
                json=body,
                headers={
                    "Content-type": content_type,
                },
                params={"key": self.api_key},
                **self.http_req_params,
            ).json()
            return "ok", response["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(response)
            # network error
            import traceback

            self.logger.warning(traceback.format_exc())
            return "err", None

    def query(self, seeds, queries):
        # Gemini does not accept a seed option
        # so we just ignore the seeds
        self.logger.info("Querying LLM model...")
        for query in queries:
            self.logger.info(query + "\n")

        responses = []
        for query in queries:
            for retry in range(self.num_retry):
                status, response = self.restful_request(query)
                if status == "ok":
                    responses.append(response)
                    break
                else:
                    if retry == self.num_retry - 1:
                        self.logger.error(
                            f"Failed to query Gemini for {self.num_retry} times. Abort!"
                        )
                        raise Exception("Failed to query Gemini")
                    else:
                        self.logger.warning(
                            "Failed to query Gemini. Sleep and retry..."
                        )
                        time.sleep(30)

        self.logger.info("LLM model responses:")
        for response in responses:
            self.logger.info(response + "\n")

        return responses


class TGIBackend:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        http_req_params: dict = {},
        max_tokens=10000,
        num_workers=4,
        model="",
        api_key="-",
        extra_args={},
    ) -> None:
        """
        Parameters
        ----------
        url
            The URL of the TGI API.
        http_req_params
            Additional HTTP request parameters.
            Common parameters include `timeout`, `proxies`, etc.
        """
        super().__init__()
        self.url = base_url + "/v1/chat/completions"
        self.http_req_params = http_req_params
        self.logger = logging.getLogger("phylox")
        self.num_workers = num_workers
        self.num_retry = 100000
        self.usage_history = []
        if num_workers > 1:
            self.pool = ThreadPoolExecutor(max_workers=num_workers)
        self.extra_args = extra_args
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def _one_restful_request(self, args):
        seed, query = args
        if len(query) > 1_000_000:
            query = query[:1_000_000]
            self.logger.warning("Query too long, truncated to 30,000 characters.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]
        data = {
            "messages": messages,
            "stream": False,
            "model": self.model,
            "seed": seed,
            "max_tokens": self.max_tokens,
        }
        data = data | self.extra_args
        for retry in range(self.num_retry):
            try:
                response = httpx.post(
                    self.url, headers=headers, json=data, **self.http_req_params
                )
                json_response = response.json()
            except Exception:
                import traceback

                self.logger.error(f"Failed to query TGI. Sleep. Retry {retry + 1}...")
                self.logger.error(traceback.format_exc())
                # network error, sleep and retry
                time.sleep(30)
                continue

            try:
                content = json_response["choices"][0]["message"]["content"]
                usage = json_response["usage"]
                return content, usage
            except Exception:
                self.logger.error(f"Failed to parse response: {json_response}")
                self.logger.error(f"{response.text}")
                self.logger.error(f"{response.json()}")

        self.logger.error(f"Failed to query TGI for {self.num_retry} times. Abort!")
        raise Exception("Failed to query TGI")

    def query(self, seeds, queries):
        self.logger.info("Querying LLM model...")
        for query in queries:
            self.logger.info(query + "\n")

        if self.num_workers > 1:
            responses = list(
                self.pool.map(self._one_restful_request, zip(seeds, queries))
            )
        else:
            responses = []
            for seed, query in zip(seeds, queries):
                responses.append(self._one_restful_request((seed, query)))

        contents, usages = zip(*responses)
        self.usage_history.append(usages)

        self.logger.info("LLM model responses:")
        for content in contents:
            self.logger.info(content + "\n")

        return contents


class HuggingfaceModel:
    """Support huggingface models"""

    def __init__(
        self,
        model,  # e.g. meta-llama/Meta-Llama-3-8B-Instruct
        device_map="auto",
        trust_remote_code=False,
        num_workers=1,
        **kwargs,
    ):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation

        self.logger = logging.getLogger("phylox")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
        llm = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                model, trust_remote_code=trust_remote_code, device_map=device_map
            )
        )
        self.chatbot = transformers.pipeline(
            "conversational",
            model=llm,
            device_map=device_map,
            tokenizer=self.tokenizer,
            trust_remote_code=trust_remote_code,
            num_workers=num_workers,
        )

    def query(self, seeds, queries):
        import torch

        # torch doesn't support setting a batch of seeds for a batch of queries
        # so we combine them into a single seed by XORing them
        seed = reduce(lambda x, y: x ^ y, seeds)
        torch.manual_seed(seed)
        self.logger.info("Querying LLM model...")
        for query in queries:
            self.logger.info(query + "\n")

        conversations = [Conversation(query) for query in queries]
        conversations = self.chatbot(
            conversations,
            top_p=0.97,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=2048,
        )
        responses = [
            conversation.messages[-1]["content"] for conversation in conversations
        ]
        self.logger.info("LLM model responses:")
        for response in responses:
            self.logger.info(response + "\n")

        return responses
