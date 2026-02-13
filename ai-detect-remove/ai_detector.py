import os

os.environ["DISABLE_TQDM"] = "1"

import numpy as np
import joblib
import torch
from transformers import BertTokenizer, BertModel
from transformers.utils import logging as hf_logging
import re

hf_logging.disable_progress_bar()


class AIDetector:
    def __init__(self, mlp_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert_model.eval()
        self.mlp_model = joblib.load(mlp_model_path)

    @staticmethod
    def split_text_into_chunks(text: str, max_words: int = 512) -> list[str]:
        words = (text or "").split()
        if not words:
            return [""]
        return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    def _predict_chunk_prob(self, chunk: str) -> float:
        """
        Predict raw AI probability for a single text chunk (0.0–1.0).
        Shared by document-level and sentence-level scoring.
        """
        inputs = self.tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.bert_model(**inputs.to(self.device))
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        prob = self.mlp_model.predict_proba([emb])[0][1]
        return float(prob)

    def predict_ai_generated_percent(self, text: str) -> float:
        chunks = self.split_text_into_chunks(text)
        probs = []

        for chunk in chunks:
            if not chunk.strip():
                continue

            probs.append(self._predict_chunk_prob(chunk))

        if not probs:
            return 0.0
        return round(float(np.mean(probs)) * 100.0, 2)

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Lightweight sentence splitter (no extra NLTK dependency here).
        Filters out very short sentences that are hard to classify.
        """
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [p.strip() for p in parts if len(p.strip().split()) >= 5]
        return sentences

    def predict_sentence_scores(self, text: str) -> list[tuple[str, float]]:
        """
        Return per-sentence AI scores (0–100) for more detailed analysis.
        """
        sentences = self._split_into_sentences(text)
        results: list[tuple[str, float]] = []

        for s in sentences:
            prob = self._predict_chunk_prob(s)
            results.append((s, round(prob * 100.0, 2)))

        return results



