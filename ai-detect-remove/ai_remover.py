import os
os.environ["DISABLE_TQDM"] = "1"

import random
import re
from dataclasses import dataclass
from typing import Optional

import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass(frozen=True)
class HumanizeConfig:
    # Maximum fraction of replaceable words to swap with synonyms
    replace_ratio: float = 0.10
    # Min word length to consider for replacement (avoid tiny/important tokens)
    min_word_len: int = 5


class AIRemover:
    """
    "AI Remove" here means rewriting/paraphrasing text to look more human.
    This is NOT guaranteed to fool all detectors, but provides a practical rewrite step.
    """

    def __init__(self):
        # NLTK resources for soft humanize
        # Newer NLTK uses 'averaged_perceptron_tagger_eng', older uses 'averaged_perceptron_tagger'
        for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"):
            try:
                nltk.data.find(pkg)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

        # Some transformers installs (like yours) don't expose the "text2text-generation"
        # pipeline task name. So we load T5 directly and run `generate()`.
        #
        # Use a smaller model to avoid Windows memory / paging-file errors.
        self.model_name = "t5-small"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _wn_pos(tag: str):
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return None

    @staticmethod
    def _is_protected_token(tok: str) -> bool:
        # keep numbers, mixed alphanumerics, emails/urls-ish, and acronyms
        if any(ch.isdigit() for ch in tok):
            return True
        if re.search(r"[@:/\\]|\.com|\.in|\.org", tok.lower()):
            return True
        if tok.isupper() and len(tok) <= 6:
            return True
        return False

    def rewrite_soft(self, text: str, cfg: HumanizeConfig = HumanizeConfig()) -> str:
        """
        Soft humanize: minimal synonym replacement to keep content very similar.
        """
        if not text or not text.strip():
            return text

        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Newer NLTK versions require explicit punkt_tab download
            for pkg in ("punkt", "punkt_tab"):
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
            try:
                tokens = word_tokenize(text)
            except Exception:
                # Fallback: simple whitespace split
                tokens = text.split()
        # POS tagging; if tagger data is missing, fall back gracefully.
        try:
            tags = pos_tag(tokens)
        except LookupError:
            for pkg in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
            try:
                tags = pos_tag(tokens)
            except Exception:
                tags = [(tok, "") for tok in tokens]

        # candidate indices (only alpha words, not protected)
        candidates = []
        for i, (tok, tag) in enumerate(tags):
            if not tok.isalpha():
                continue
            if len(tok) < cfg.min_word_len:
                continue
            if self._is_protected_token(tok):
                continue
            if self._wn_pos(tag) is None:
                continue
            candidates.append(i)

        if not candidates:
            return text

        k = max(1, int(len(candidates) * cfg.replace_ratio))
        chosen = set(random.sample(candidates, k=min(k, len(candidates))))

        out = []
        for i, (tok, tag) in enumerate(tags):
            if i not in chosen:
                out.append(tok)
                continue

            wn_tag = self._wn_pos(tag)
            replaced = tok
            try:
                syns = wordnet.synsets(tok, pos=wn_tag)
                for s in syns:
                    for lemma in s.lemmas():
                        name = lemma.name().replace("_", " ")
                        if name.lower() != tok.lower() and name.isalpha():
                            replaced = name
                            break
                    if replaced.lower() != tok.lower():
                        break
            except Exception:
                replaced = tok

            # preserve capitalization
            if tok.istitle():
                replaced = replaced.title()
            out.append(replaced)

        # Join with spaces, then fix common tokenization spacing
        result = " ".join(out)
        result = re.sub(r"\s+([.,!?;:])", r"\1", result)
        result = re.sub(r"\(\s+", "(", result)
        result = re.sub(r"\s+\)", ")", result)
        return result

    def rewrite_strong(self, text: str, max_chars: int = 6000) -> str:
        """
        Strong rewrite using T5 (can change wording more).
        max_chars limits work to keep UI responsive.
        """
        if not text or not text.strip():
            return text

        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars]

        # Simple paragraph-based chunking (safer than huge single prompt)
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        out = []
        for p in paragraphs:
            prompt = "paraphrase: " + p

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=180,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True,
                )

            out.append(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))

        return "\n\n".join(out)

    def rewrite(self, text: str, strength: str = "soft", ai_score_before: Optional[float] = None) -> str:
        """
        Adaptive rewrite:
        - If user selects "strong" OR the detector score is very high (>= 90),
          run a strong T5 rewrite, then optionally add a heavier synonym pass.
        - For "soft" mode with high AI score, increase synonym replacements a bit.
        """
        high_risk = ai_score_before is not None and ai_score_before >= 90.0

        if strength == "strong" or high_risk:
            # Strong paraphrase first
            rewritten = self.rewrite_strong(text)

            # For extremely AI-looking text, add an additional, slightly stronger
            # synonym layer to diversify wording further.
            if high_risk:
                cfg = HumanizeConfig(replace_ratio=0.20, min_word_len=4)
                rewritten = self.rewrite_soft(rewritten, cfg=cfg)
            return rewritten

        # "soft" mode: on moderately high AI scores, nudge config a bit stronger
        if ai_score_before is not None and ai_score_before >= 80.0:
            cfg = HumanizeConfig(replace_ratio=0.15, min_word_len=4)
            return self.rewrite_soft(text, cfg=cfg)

        return self.rewrite_soft(text)


