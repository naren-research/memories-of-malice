import os
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Big5InferenceModel:
    """
    Inference wrapper around a pre-trained Big Five model from the Hugging Face Hub.

    - Defaults to Minej/bert-base-personality, which documents label order as:
      [Extroversion, Neuroticism, Agreeableness, Conscientiousness, Openness].
    - Returns a canonical mapping with keys:
      ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism'].
    - Ensures outputs are in [0, 1]. If the model is multi-label classification, applies sigmoid.
      If it's regression, clamps to [0, 1]. If unsure, auto-detects and applies sigmoid when raw
      logits look like unbounded values.
    """

    CANONICAL_TRAITS = [
        "agreeableness",
        "openness",
        "conscientiousness",
        "extraversion",
        "neuroticism",
    ]

    def __init__(
        self,
        model_name: str = "Minej/bert-base-personality",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        elif self.device == "cuda":
            # Safe default for inference on modern GPUs
            model_kwargs["torch_dtype"] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

        # Determine problem type and label order
        self.problem_type: Optional[str] = getattr(self.model.config, "problem_type", None)
        self.trait_order = self._determine_trait_order()

    def _determine_trait_order(self) -> List[str]:
        name = self.model_name.lower()
        # Known orders for popular models
        if "minej/bert-base-personality" in name:
            # Documented in model card
            return [
                "extraversion",
                "neuroticism",
                "agreeableness",
                "conscientiousness",
                "openness",
            ]
        if "vladinc/bigfive-regression-model" in name:
            # Common OCEAN order used in many datasets (best-effort assumption)
            return [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]

        # Try to infer from id2label if present and descriptive
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) == 5:
            # Convert to list in index order 0..4
            ordered = [id2label.get(str(i), id2label.get(i, f"LABEL_{i}")) for i in range(5)]
            lower = [str(x).lower() for x in ordered]
            if any(any(t in lbl for t in ["open", "openness"]) for lbl in lower):
                # Normalize labels to canonical keys when possible
                norm = []
                for lbl in lower:
                    if "agree" in lbl:
                        norm.append("agreeableness")
                    elif "open" in lbl:
                        norm.append("openness")
                    elif "consci" in lbl:
                        norm.append("conscientiousness")
                    elif "extra" in lbl or "extro" in lbl:
                        norm.append("extraversion")
                    elif "neuro" in lbl:
                        norm.append("neuroticism")
                    else:
                        norm.append(lbl)
                if len(set(norm)) == 5:
                    return norm

        # Fallback to OCEAN order
        return [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to probabilities in [0,1]."""
        if self.problem_type == "multi_label_classification":
            probs = torch.sigmoid(logits)
        elif self.problem_type == "regression":
            probs = logits
        else:
            # Auto-detect: if values appear unbounded, apply sigmoid
            with torch.no_grad():
                min_v = logits.min().item()
                max_v = logits.max().item()
            if min_v < 0.0 or max_v > 1.5:
                probs = torch.sigmoid(logits)
            else:
                probs = logits
        # Ensure [0,1]
        return probs.clamp(0.0, 1.0)

    def score_texts(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, float]]:
        """
        Score a list of texts and return a list of dictionaries mapping canonical trait names
        to values in [0,1]. Order-independent for callers.
        """
        results: List[Dict[str, float]] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            probs = self._logits_to_probs(logits).detach().cpu().tolist()

            for vec in probs:
                trait_map = {t: float(v) for t, v in zip(self.trait_order, vec)}
                canonical = {
                    "agreeableness": trait_map.get("agreeableness", 0.0),
                    "openness": trait_map.get("openness", 0.0),
                    "conscientiousness": trait_map.get("conscientiousness", 0.0),
                    "extraversion": trait_map.get("extraversion", 0.0),
                    "neuroticism": trait_map.get("neuroticism", 0.0),
                }
                results.append(canonical)
        return results

    def score_text(self, text: str) -> Dict[str, float]:
        return self.score_texts([text])[0]

    def score_openness(self, texts: List[str], batch_size: int = 64) -> List[float]:
        scores = self.score_texts(texts, batch_size=batch_size)
        return [float(s["openness"]) for s in scores]


if __name__ == "__main__":
    model = Big5InferenceModel()
    demo = [
        "I love exploring new ideas, traveling to different places, and experimenting with creative hobbies.",
        "I prefer routine and tradition, and I rarely seek out new experiences.",
    ]
    scores = model.score_texts(demo)
    for t, s in zip(demo, scores):
        print("TEXT:", t)
        print("SCORES:", s)
        print("openness:", s["openness"])
