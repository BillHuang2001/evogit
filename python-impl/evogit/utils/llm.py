"""
Utility functions for the controlling LLM models.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
import time
from functools import reduce
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import httpx


class TGIBackend:
    def __init__(
        self,
        endpoint,
        max_tokens=10000,
        num_workers=4,
        model="",
        api_key="-",
        top_p=0.9,
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
        self.logger = logging.getLogger("evogit")
        self.num_workers = num_workers
        self.num_retry = 100000
        self.usage_history = []
        if num_workers > 1:
            self.pool = ThreadPoolExecutor(max_workers=num_workers)
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        if not endpoint.endswith("/"):
            endpoint = endpoint + "/"
        self.client = ChatCompletionsClient(
            endpoint=endpoint + "openai/deployments/" + model,
            credential=AzureKeyCredential(api_key),
            top_p=top_p,
        )

    def _one_restful_request(self, args):
        seed, query = args
        # the context window is 128k, and 4 chars is around 1 token
        if len(query) > 400_000:
            query = query[:400_000]
            self.logger.warning("Query too long, truncated to 400,000 characters.")

        for retry in range(self.num_retry):
            try:
                response = self.client.complete(
                    messages=[
                        UserMessage(content=query),
                    ],
                    max_tokens=self.max_tokens,
                    model=self.model,
                )
                content = response.choices[0].message.content
                usage = response.usage
                return content, usage
            except Exception as e:
                self.logger.warning(f"Failed to query TGI: {e}")
                time.sleep(60)

        self.logger.error(f"Failed to query TGI for {self.num_retry} times. Abort!")
        raise Exception("Failed to query TGI")

    def query(self, seeds, queries):
        if len(queries) == 0:
            self.logger.warning("No queries to process.")
            return []

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
