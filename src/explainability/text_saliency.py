"""
TruthLens AI - Text Saliency Attribution
Computes word and token-level gradient attribution for DistilBERT NLP inputs.
Identifies which specific words swayed the model toward REAL, FAKE, or UNCERTAIN.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from src.preprocessing.text_preprocessor import TextPreprocessor


class TextSaliencyExplainer:
    """
    Computes gradient-based token importance scores for DistilBERT transformer models.
    """
    def __init__(self, model: nn.Module, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        self.model = model
        self.model.eval()
        self.text_proc = TextPreprocessor()
        self.tokenizer = tokenizer or self.text_proc.tokenizer

        # Access DistilBERT word embedding layer
        if hasattr(model, "transformer") and hasattr(model.transformer, "embeddings"):
            self.embedding_layer = model.transformer.embeddings.word_embeddings
        elif hasattr(model, "text_model") and hasattr(model.text_model.transformer, "embeddings"):
            self.embedding_layer = model.text_model.transformer.embeddings.word_embeddings
        else:
            raise ValueError("Could not locate word_embeddings layer on the provided model.")

    def explain(
        self,
        text: str,
        target_class: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Tokenizes text, computes backprop gradient norm at token embeddings,
        and returns list of tokens with normalized importance scores.
        """
        cleaned = self.text_proc.clean_text(text)
        tokenized = self.tokenizer(cleaned, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Convert tokens to readable words
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Hook to capture gradients at embedding layer
        embeddings = self.embedding_layer(input_ids)
        embeddings.retain_grad()

        # Forward pass through transformer using the embeddings
        if hasattr(self.model, "transformer"):
            outputs = self.model.transformer(inputs_embeds=embeddings, attention_mask=attention_mask)
            cls_token = outputs.last_hidden_state[:, 0, :]
            logits = self.model.classifier(cls_token)
        else:
            raise NotImplementedError("Direct forward with inputs_embeds on custom wrapper.")

        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward()

        # Compute L2 norm of gradients per token position
        grads = embeddings.grad[0]  # Shape: (seq_len, embedding_dim)
        importance_scores = torch.norm(grads, dim=1).detach().cpu().numpy()

        # Normalize scores to [0, 1]
        max_score = importance_scores.max()
        if max_score > 0:
            importance_scores /= max_score

        results = []
        for token, imp in zip(tokens, importance_scores):
            # Skip special tokens [CLS], [SEP], [PAD] in presentation attribution
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            results.append({
                "token": token,
                "importance": float(np.round(imp, 4))
            })

        return results
