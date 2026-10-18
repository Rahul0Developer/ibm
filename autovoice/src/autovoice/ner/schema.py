"""NER label schema: entity types, BIO tag set and token/label utilities.

Entity types (Phase 3):
    COMPONENT            - vehicle subsystem/part named in the narrative
                           (e.g. "brake caliper", "12-volt battery").
    SYMPTOM              - failure symptom described by the complainant
                           (e.g. "stalled", "grinding noise", "warning light").
    VEHICLE_PART_LOCATION- optional third type ("front left", "under the hood");
                           enabled only if it does not hurt gold-set F1
                           (decision recorded in docs/model_card.md).

The model uses IOB2 tags: B-<TYPE>, I-<TYPE>, O.  Sub-word tokens beyond the
first of a word get label -100 (ignored by the HF Trainer loss).
"""

from __future__ import annotations

from enum import StrEnum

IGNORE_LABEL_ID = -100


class EntityType(StrEnum):
    """Top-level entity categories (not the BIO tags themselves)."""

    COMPONENT = "COMPONENT"
    SYMPTOM = "SYMPTOM"
    LOCATION = "VEHICLE_PART_LOCATION"


#: Canonical entity types; drop LOCATION here to disable it everywhere.
ACTIVE_ENTITY_TYPES: tuple[EntityType, ...] = (EntityType.COMPONENT, EntityType.SYMPTOM)


def bio_labels(entity_types: tuple[EntityType, ...] = ACTIVE_ENTITY_TYPES) -> list[str]:
    """Return the ordered BIO tag list (also used to build the label2id map).

    Args:
        entity_types: Entity types to include.

    Returns:
        List like ``["O", "B-COMPONENT", "I-COMPONENT", ...]``.
    """
    labels = ["O"]
    for et in entity_types:
        labels += [f"B-{et}", f"I-{et}"]
    return labels


def make_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build deterministic id<->tag maps from an ordered tag list."""
    return {l: i for i, l in enumerate(labels)}, {i: l for i, l in enumerate(labels)}


def spans_to_bio(words: list[str], spans: list[tuple[int, int, str]], labels: list[str]) -> list[str]:
    """Convert character-offset spans into per-word BIO tags.

    Overlap policy: a word is tagged if any character of the word falls inside
    a span; later spans win on conflict (spans should be non-overlapping from
    the weak labeler, which resolves overlaps longest-match-first).

    Args:
        words: whitespace-split tokens with their char offsets recomputed here.
        spans: list of (start_char, end_char_exclusive, BIO_TAG e.g. 'B-SYMPTOM').
        labels: valid tag vocabulary (unknown tags raise ValueError).

    Returns:
        One tag per word.
    """
    tags = ["O"] * len(words)
    valid = set(labels)
    joined = " ".join(words)
    # recompute word offsets against the single-space join
    pos, offsets = 0, []
    for w in words:
        while pos < len(joined) and joined[pos].isspace():
            pos += 1
        offsets.append((pos, pos + len(w)))
        pos += len(w)
    for start, end, tag in spans:
        if tag not in valid:
            raise ValueError(f"Unknown BIO tag {tag!r}; valid: {sorted(valid)}")
        covered = [i for i, (ws, we) in enumerate(offsets) if ws < end and we > start]
        for k, i in enumerate(covered):
            tags[i] = tag if k == 0 else ("I-" + tag[2:])
    return tags


def bio_to_spans(
    words: list[str], tags: list[str], min_confidence_words: int = 1
) -> list[tuple[int, int, str, int]]:
    """Decode a per-word BIO sequence into (word_start, word_end_excl, type, n_words).

    Used by evaluation and by the API span post-processing; I-without-B chunks
    are treated as entities too (common with weak models).
    """
    spans: list[tuple[int, int, str, int]] = []
    i = 0
    n = len(words)
    while i < n:
        tag = tags[i]
        if tag == "O" or not tag[:2] in ("B-", "I-"):
            i += 1
            continue
        etype = tag[2:]
        j = i + 1
        while j < n and tags[j] in (f"I-{etype}",):
            j += 1
        if j - i >= min_confidence_words:
            spans.append((i, j, etype, j - i))
        i = j
    return spans
