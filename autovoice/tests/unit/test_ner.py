"""Unit tests for NER label alignment, BIO conversion and span decoding (Phase 3).

No model download required - these test the pure functions that make or break
token-classification correctness.
"""

from __future__ import annotations

import pytest

from autovoice.ner.infer import decode_word_tags
from autovoice.ner.schema import (
    IGNORE_LABEL_ID,
    bio_labels,
    bio_to_spans,
    make_label_maps,
    spans_to_bio,
)
from autovoice.ner.train import align_word_labels
from autovoice.ner.weak_labeler import tag_text

LABELS = bio_labels()
L2I, _ = make_label_maps(LABELS)


class TestSchema:
    def test_bio_labels_order(self):
        assert LABELS == ["O", "B-COMPONENT", "I-COMPONENT", "B-SYMPTOM", "I-SYMPTOM"]

    def test_spans_to_bio_multiword_entity(self):
        words = "the brake caliper failed".split()
        # span covers "brake caliper" chars 4..18
        tags = spans_to_bio(words, [(4, 18, "B-COMPONENT")], LABELS)
        assert tags == ["O", "B-COMPONENT", "I-COMPONENT", "O"]

    def test_spans_to_bio_rejects_unknown_tag(self):
        with pytest.raises(ValueError):
            spans_to_bio(["x"], [(0, 1, "B-NOPE")], LABELS)

    def test_bio_to_spans_roundtrip(self):
        words = "engine stalled badly".split()
        tags = ["B-COMPONENT", "B-SYMPTOM", "O"]
        spans = bio_to_spans(words, tags)
        assert (0, 1, "COMPONENT", 1) in spans
        assert (1, 2, "SYMPTOM", 1) in spans

    def test_bio_to_spans_i_without_b(self):
        spans = bio_to_spans(["a", "b"], ["O", "I-SYMPTOM"])
        assert spans == [(1, 2, "SYMPTOM", 1)]


class TestAlignment:
    def test_first_subtoken_only(self):
        # word_ids for [CLS] en ##gi ##ne [SEP]: None,0,0,0,1,None style
        word_ids = [None, 0, 0, 1, 2, None]
        words = ["engine", "is", "dead"]
        tags = ["B-COMPONENT", "O", "B-SYMPTOM"]
        labels = align_word_labels(words, tags, word_ids, L2I)
        assert labels[0] == IGNORE_LABEL_ID          # [CLS]
        assert labels[1] == L2I["B-COMPONENT"]       # first sub-token of word 0
        assert labels[2] == IGNORE_LABEL_ID          # continuation
        assert labels[3] == L2I["O"]                 # word 1
        assert labels[4] == L2I["B-SYMPTOM"]         # word 2
        assert labels[5] == IGNORE_LABEL_ID          # [SEP]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            align_word_labels(["a", "b"], ["O"], [0, 1], L2I)


class TestDecode:
    def test_decode_word_tags_first_wins(self):
        tag_ids = [0, 1, 1, 3, 0]
        word_ids = [None, 0, 0, 1, 2]
        id2l = {i: l for l, i in L2I.items()}
        out = decode_word_tags(tag_ids, word_ids, id2l, n_words=3)
        assert out == ["B-COMPONENT", "B-SYMPTOM", "O"]


class TestWeakLabeler:
    def test_symptom_and_component_detected(self):
        spans = tag_text("The brake pads started grinding and the warning light came on.")
        labels = {t for _, _, t in spans}
        assert "B-COMPONENT" in labels and "B-SYMPTOM" in labels

    def test_longest_match_wins(self):
        spans = tag_text("check the brake caliper now")
        matched = [(s, e) for s, e, _ in spans]
        text = "check the brake caliper now"
        assert any(text[s:e] == "brake caliper" for s, e in matched)

    def test_no_overlapping_spans(self):
        spans = tag_text("battery not charging and engine overheating")
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a, b = spans[i], spans[j]
                assert b[0] >= a[1] or a[0] >= b[1]
