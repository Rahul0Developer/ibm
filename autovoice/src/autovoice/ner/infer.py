"""NER inference: local dir or HF Hub, char offsets, confidence, sliding window.

Used by the FastAPI service and the RAG enrich node.  Long texts are split
into overlapping word-chunks (window + stride) so entities near boundaries
are not lost; duplicate spans from overlap are de-duplicated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from autovoice.logging_conf import get_logger
from autovoice.ner.schema import bio_labels, bio_to_spans

log = get_logger(__name__)

WINDOW_WORDS = 100
STRIDE_WORDS = 80


def decode_word_tags(tag_ids: list[int], word_ids: list[int | None],
                     id2label: dict[int, str], n_words: int) -> list[str]:
    """Collapse sub-token argmax labels to one BIO tag per word (first sub-token wins)."""
    tags = ["O"] * n_words
    seen: set[int] = set()
    for tid, wid in zip(tag_ids, word_ids, strict=False):
        if wid is None or wid in seen or wid >= n_words:
            continue
        seen.add(wid)
        tag = id2label.get(int(tid), "O")
        # continuation sub-tokens of an I- word shouldn't flip B->I etc.; first wins
        tags[wid] = tag
    return tags


@dataclass
class Entity:
    """A predicted entity with char offsets and mean softmax confidence."""

    text: str
    label: str
    start: int
    end: int
    score: float

    def to_dict(self) -> dict:
        return {"text": self.text, "label": self.label, "start": self.start,
                "end": self.end, "score": round(self.score, 4)}


class NERInferencer:
    """Lazy-loading NER model wrapper (thread-unsafe by design; API loads once)."""

    def __init__(self, source: str):
        """``source`` is a local model directory or a Hugging Face Hub repo id."""
        self.source = source
        self._tokenizer = None
        self._model = None
        self._labels = bio_labels()

    @property
    def labels(self) -> list[str]:
        """BIO tag vocabulary of the loaded model (falls back to schema default)."""
        return self._labels

    def load(self) -> None:
        """Load tokenizer + model (CPU). Idempotent."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        log.info("ner.inferencer.loading", source=self.source)
        self._tokenizer = AutoTokenizer.from_pretrained(self.source)
        self._model = AutoModelForTokenClassification.from_pretrained(self.source)
        self._model.eval()
        cfg_labels = getattr(self._model.config, "labels", None)
        if cfg_labels:
            self._labels = [str(l) for l in cfg_labels]
        self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        self._torch = torch

    def warmup(self) -> None:
        """Run one dummy prediction so Cloud Run cold starts include model init."""
        self.predict("engine stalled while driving")

    def predict(self, text: str) -> list[Entity]:
        """Return entities with char offsets + confidences for arbitrary-length text."""
        self.load()
        words = text.split()
        if not words:
            return []
        # char offset of each whitespace word in the original text
        offsets, pos = [], 0
        for w in words:
            idx = text.index(w, pos)
            offsets.append((idx, idx + len(w)))
            pos = idx + len(w)

        all_tags: list[str] = ["O"] * len(words)
        all_scores: list[float] = [0.0] * len(words)
        starts = range(0, max(1, len(words)), STRIDE_WORDS) if len(words) > WINDOW_WORDS else [0]
        for s in starts:
            chunk = words[s:s + WINDOW_WORDS]
            if not chunk:
                break
            enc = self._tokenizer(chunk, truncation=True, is_split_into_words=True,
                                  return_tensors="pt", max_length=512)
            with self._torch.no_grad():
                outputs = self._model(**enc)
                probs = self._torch.softmax(outputs.logits, dim=-1)[0]
            tag_ids = probs.argmax(dim=-1).tolist()
            confs = probs.max(dim=-1).values.tolist()
            word_ids = enc.word_ids()
            n_chunk = len(chunk)
            for i, (tid, wid) in enumerate(zip(tag_ids, word_ids, strict=False)):
                if wid is None or wid >= n_chunk:
                    continue
                gi = s + wid
                if gi >= len(words):
                    break
                tag = self._id2label.get(int(tid), "O")
                # keep the higher-confidence assignment across windows
                if tag != "O" and all_tags[gi] == "O":
                    all_tags[gi] = tag
                    all_scores[gi] = float(confs[i])
                elif tag != "O" and confs[i] > all_scores[gi]:
                    all_tags[gi] = tag
                    all_scores[gi] = float(confs[i])

        entities: list[Entity] = []
        for ws, we, etype, _n in bio_to_spans(words, all_tags):
            ent_text = " ".join(words[ws:we])
            entities.append(Entity(
                text=ent_text, label=etype,
                start=offsets[ws][0], end=offsets[we - 1][1],
                score=sum(all_scores[ws:we]) / (we - ws),
            ))
        return entities


def load_reference_config(model_dir: str | Path) -> dict:
    """Read the training config saved next to the weights (for /v1/model-info)."""
    p = Path(model_dir) / "ner_train_config.json"
    return json.loads(p.read_text()) if p.exists() else {}
