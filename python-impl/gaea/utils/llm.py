"""
Utility functions for the controlling LLM models.
"""

import logging
import time
from functools import reduce

import httpx
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation


class GeminiBackend:
    def __init__(self, api_key: str, http_req_params: dict = {}) -> None:
        """
        Parameters
        ----------
        api_key
            The API key for the Gemini API.
        http_req_params
            Additional HTTP request parameters.
            Common parameters include `timeout`, `proxies`, etc.
        """
        super().__init__()
        self.api_key = api_key
        self.http_req_params = http_req_params
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.logger = logging.getLogger("main")
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


class HuggingfaceModel:
    """Support huggingface models"""

    def __init__(
        self,
        model,  # e.g. meta-llama/Meta-Llama-3-8B-Instruct
        device_map="auto",
        trust_remote_code=False,
        num_workers=1,
    ):
        self.logger = logging.getLogger("main")

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
