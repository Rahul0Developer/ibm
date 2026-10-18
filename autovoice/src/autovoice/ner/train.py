"""Fine-tune a DistilBERT token classifier for AutoVoice NER.

Runs anywhere (CPU works on the small dataset) and is the engine behind
``notebooks/02_train_ner_colab.ipynb``.  Key correctness detail: **sub-word
label alignment** - only the first sub-token of each word receives its BIO
label id; continuation sub-tokens and special tokens get ``-100`` and are
ignored by the cross-entropy loss (alignment uses ``word_ids()``, which is
offset-independent and correct for WordPiece).

Determinism: seeds fixed via YAML config + ``set_seed``; the resolved config
is saved next to the weights.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from autovoice.config import DATA_LABELED_DIR, MODELS_DIR
from autovoice.logging_conf import get_logger
from autovoice.ner.schema import IGNORE_LABEL_ID, bio_labels, make_label_maps

log = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/ner.yaml")


@dataclass
class NerTrainConfig:
    """Training hyper-parameters; loaded from configs/ner.yaml."""

    model_name: str = "distilbert-base-uncased"
    entity_types: list[str] = field(default_factory=lambda: ["COMPONENT", "SYMPTOM"])
    silver_path: str = str(DATA_LABELED_DIR / "silver_ner.jsonl")
    output_dir: str = str(MODELS_DIR / "ner-distilbert")
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 16
    num_epochs: int = 4
    max_seq_len: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    max_train_examples: int | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "NerTrainConfig":
        p = path or DEFAULT_CONFIG_PATH
        if p.exists():
            return cls(**yaml.safe_load(p.read_text()))
        log.warning("ner.config_missing_using_defaults", path=str(p))
        return cls()


def align_word_labels(words: list[str], word_tags: list[str], word_ids: list[int | None],
                      label2id: dict[str, int]) -> list[int]:
    """Pure sub-word alignment function (unit-testable).

    Args:
        words: whitespace tokens (only used for length validation).
        word_tags: one BIO tag per word.
        word_ids: the ``word_ids()`` list from a fast tokenizer encoding that
            includes special tokens (``None`` at special positions).
        label2id: BIO tag -> id map.

    Returns:
        One label id per encoding position; -100 for continuations/specials.
    """
    if len(words) != len(word_tags):
        raise ValueError("words and word_tags must have equal length")
    labels: list[int] = []
    prev_wid: int | None = None
    for wid in word_ids:
        if wid is None or wid == prev_wid or wid >= len(word_tags):
            labels.append(IGNORE_LABEL_ID)
        else:
            labels.append(label2id[word_tags[wid]])
        prev_wid = wid
    return labels


def build_dataset_records(jsonl_path: Path) -> list[dict]:
    """Read span JSONL -> list of {words, tags} using schema.spans_to_bio."""
    from autovoice.ner.schema import spans_to_bio

    records = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        words = obj["text"].split()
        if not words:
            continue
        spans = [(s["start"], s["end"], s["label"]) for s in obj.get("spans", [])]
        tags = spans_to_bio(words, spans, bio_labels())
        records.append({"words": words, "tags": tags})
    return records


def train(cfg: NerTrainConfig) -> Path:
    """Fine-tune the token classifier; returns the output model directory."""
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    labels = bio_labels(tuple(cfg.entity_types))
    label2id, id2label = make_label_maps(labels)

    records = build_dataset_records(Path(cfg.silver_path))
    if cfg.max_train_examples:
        records = records[: cfg.max_train_examples]
    if not records:
        raise SystemExit(f"No training records found in {cfg.silver_path}; run `make data` first.")
    log.info("ner.train_data", n=len(records))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    # Pre-tokenize once (dataset is small); alignment via word_ids().
    features = []
    for rec in records:
        enc = tokenizer(rec["words"], truncation=True, is_split_into_words=True,
                        max_length=cfg.max_seq_len)
        feat_labels = align_word_labels(rec["words"], rec["tags"], enc.word_ids(), label2id)
        pad = len(enc["input_ids"]) - len(feat_labels)
        feat_labels += [IGNORE_LABEL_ID] * max(0, pad)
        features.append({"input_ids": enc["input_ids"], "labels": feat_labels[: len(enc["input_ids"])]})

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, feats):
            self.feats = feats

        def __len__(self):
            return len(self.feats)

        def __getitem__(self, i):
            return self.feats[i]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, mask, labs = [], [], []
        for b in batch:
            n = max_len - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * n)
            mask.append([1] * len(b["input_ids"]) + [0] * n)
            labs.append(b["labels"] + [IGNORE_LABEL_ID] * n)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask),
                "labels": torch.tensor(labs)}

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        seed=cfg.seed,
        data_seed=cfg.seed,
        logging_steps=25,
        save_strategy="epoch",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ListDataset(features),
                      data_collator=collate)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    Path(cfg.output_dir, "ner_train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log.info("ner.model_saved", path=cfg.output_dir)
    return Path(cfg.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AutoVoice NER model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parsed = parser.parse_args()
    train(NerTrainConfig.load(parsed.config))


if __name__ == "__main__":
    main()
