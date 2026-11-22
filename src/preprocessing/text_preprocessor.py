"""
TruthLens AI - Text Preprocessing Pipeline
Handles text normalization, cleaning, and tokenization for transformer models (DistilBERT).
"""

import re
from typing import Dict, List, Union, Any
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TextPreprocessor:
    """
    Standardizes and tokenizes caption/text inputs for TruthLens NLP models.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans and normalizes raw textual input:
        - Decodes escape sequences
        - Removes redundant whitespace and unwanted control characters
        - Preserves sentence structure and punctuation critical for semantics
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Remove control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        # Collapse multiple spaces, tabs, and newlines into single space
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, texts: Union[str, List[str]], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Cleans and tokenizes single string or batch of strings into PyTorch tensors.
        Returns:
            Dict containing 'input_ids' and 'attention_mask'.
        """
        if isinstance(texts, str):
            cleaned = [self.clean_text(texts)]
        else:
            cleaned = [self.clean_text(t) for t in texts]

        encoding = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"]
        }
