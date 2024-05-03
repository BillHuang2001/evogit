"""
Utility functions for the controlling LLM models.
"""

import logging
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
)


class HuggingfaceModel:
    """Support huggingface models"""

    def __init__(
        self,
        model,  # e.g. meta-llama/Meta-Llama-3-8B-Instruct
        device_map="auto",
        trust_remote_code=False,
        num_workers=1,
    ):
        import torch

        tokenizer = AutoTokenizer.from_pretrained(
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
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            num_workers=num_workers,
        )

    def query(self, seeds, queries):
        conversations = [Conversation(query) for query in queries]
        conversations = self.chatbot(conversations)
        responses = [
            conversation.messages[-1]["content"] for conversation in conversations
        ]
        return responses
