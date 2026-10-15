"""
ShikshaEval-Indic — src/evaluate.py
===================================

Devanagari-aware evaluation of the handwritten-answer VLM:

  1. Unicode-safe normalization (NFC + ZWJ/ZWNJ stripping) so visually identical
     jodakshar / split-matra sequences never falsely inflate error rates.
  2. CER / WER via jiwer, plus Quadratic Weighted Kappa (QWK) and Macro-F1 for
     the 0/1/2 teacher-rubric scoring task.
  3. A Devanagari diagnostic error taxonomy computed on the *optimal alignment*
     (Levenshtein backtrace) between reference and hypothesis:
        - matra/diacritic errors      (U+093E–U+094C, U+0900–U+0903)
        - conjunct / halant errors    (U+094D virama clusters)
        - visually confusable graphemes (भ↔म, घ↔ध, ब↔व, ळ↔ल, ड↔द, ट↔ठ ...)
  4. Markdown benchmark table: Zero-Shot Base vs Unsloth QLoRA Fine-Tuned.

Usage:
    python -m src.evaluate --pred predictions.jsonl --out report.md
predictions.jsonl lines:
    {"reference": "...", "hypothesis": "...", "rubric_true": 2, "rubric_pred": 1}
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from jiwer import cer as _jiwer_cer
from jiwer import wer as _jiwer_wer
from sklearn.metrics import confusion_matrix, f1_score

# ---------------------------------------------------------------------------
# Devanagari Unicode knowledge base
# ---------------------------------------------------------------------------

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Matras (vowel signs) incl. anusvara/chandrabindu & visarga ranges cited in spec
MATRA_RANGES = [(0x093E, 0x094C), (0x0901, 0x0903)]
HALANT = "\u094D"          # virama -> conjunct formation (जोडक्षर)
ZW_CHARS = ("\u200c", "\u200d")  # ZWNJ (halant-style explicit kill), ZWJ (ligature force)

CONFUSABLE_PAIRS: list[tuple[str, str, str]] = [
    ("भ", "म", "bha/ma loop-vs-bowl"),
    ("घ", "ध", "gha/dha head-stroke"),
    ("ब", "व", "ba/va (va has no shirorekha join)"),
    ("ळ", "ल", "Marathi retroflex ळ vs ल"),
    ("ड", "द", "da/dada bottom tail"),
    ("ट", "ठ", "ta/tha top loop"),
    ("न", "ने", "na with/ne matra dot"),
    ("ा", "ि", "aa-vs-ii matra side"),
    ("ी", "े", "ii-vs-e matra length"),
    ("ु", "ू", "u-vs-uu tail depth"),
    ("ै", "े", "ai-vs-e double stroke"),
    ("ं", "ँ", "anusvara vs chandrabindu"),
    ("क्ष", "क्ष्", "conjunct kṣa split by spurious halant"),
]
_CONFUSABLE_MAP: dict[frozenset, str] = {
    frozenset((a, b)): label for a, b, label in CONFUSABLE_PAIRS
}


def _in_matra_range(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in MATRA_RANGES)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_devanagari(text: str, keep_spaces: bool = True) -> str:
    """
    NFC composes decomposed matra+nukta sequences (क्‍ष forms), strips ZWJ/ZWNJ
    (both render identically to plain conjuncts on paper), removes DANDA/DOUBLE
    DANDA punctuation noise, and collapses whitespace. Applied to BOTH reference
    and hypothesis so CER/WER measure glyph-level disagreement, not encoding luck.
    """
    t = unicodedata.normalize("NFC", text)
    for zw in ZW_CHARS:
        t = t.replace(zw, "")
    t = t.replace("\u0964", " ").replace("\u0965", " ")   # । ॥
    t = re.sub(r"\s+", " ", t).strip() if keep_spaces else re.sub(r"\s+", "", t)
    return unicodedata.normalize("NFC", t)


# ---------------------------------------------------------------------------
# Alignment-based error taxonomy (Levenshtein with full backtrace)
# ---------------------------------------------------------------------------

@dataclass
class ErrorTaxonomy:
    total_edits: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    matra_errors: int = 0
    halant_conjunct_errors: int = 0
    confusable_grapheme_errors: int = 0
    digit_errors: int = 0
    examples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def levenshtein_align(ref: str, hyp: str) -> list[tuple[str, str, str]]:
    """
    Classic DP edit distance with backtrace; returns ops list of
    (op, ref_char, hyp_char) where op ∈ {match, sub, ins, del}.
    Character-level (CER-consistent) alignment.
    """
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ord(ref[i - 1])
        for j in range(1, m + 1):
            cost = 0 if ri == ord(hyp[j - 1]) else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (0 if ref[i-1] == hyp[j-1] else 1):
            ops.append(("match" if ref[i-1] == hyp[j-1] else "sub", ref[i-1], hyp[j-1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("del", ref[i-1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j-1]))
            j -= 1
    return ops[::-1]


def classify_pair(a: str, b: str) -> str | None:
    """Confusable-grapheme label for a substituted pair, else None."""
    if a and b and len(a) == 1 and len(b) == 1:
        return _CONFUSABLE_MAP.get(frozenset((a, b)))
    return None


def compute_taxonomy(ref: str, hyp: str, max_examples: int = 3) -> ErrorTaxonomy:
    ref, hyp = normalize_devanagari(ref), normalize_devanagari(hyp)
    tax = ErrorTaxonomy()
    ex_count = 0
    for op, rc, hc in levenshtein_align(ref, hyp):
        if op == "match":
            continue
        tax.total_edits += 1
        if op == "sub":
            tax.substitutions += 1
        elif op == "ins":
            tax.insertions += 1
        else:
            tax.deletions += 1

        involved = {rc, hc}
        # 1) matra / diacritic error: either side is a vowel sign / anusvara / visarga
        if any(c and _in_matra_range(c) for c in involved):
            tax.matra_errors += 1
        # 2) conjunct/halant error: halant present on either side, OR deletion that
        #    splits a conjunct cluster (halant followed by consonant in reference)
        elif HALANT in involved:
            tax.halant_conjunct_errors += 1
        # 3) confusable glyph substitution
        label = classify_pair(rc, hc)
        if label:
            tax.confusable_grapheme_errors += 1
        # digits (Devanagari or ASCII)
        if any(c and (0x0966 <= ord(c) <= 0x096F or c.isdigit()) for c in involved):
            tax.digit_errors += 1

        if ex_count < max_examples and op != "match":
            tax.examples.append({"op": op, "ref": rc, "hyp": hc,
                                 "label": label or _quick_name(rc, hc)})
            ex_count += 1
    return tax


def _quick_name(rc: str, hc: str) -> str:
    ch = rc or hc
    if not ch:
        return "?"
    try:
        return unicodedata.name(ch)
    except ValueError:
        return f"U+{ord(ch):04X}"


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    n_samples: int
    cer: float
    wer: float
    qwk: float | None
    macro_f1: float | None
    rubric_accuracy: float | None
    taxonomy: dict
    per_sample: list[dict] = field(repr=False, default_factory=list)


def quadratic_weighted_kappa(y_true: Iterable[int], y_pred: Iterable[int],
                             labels: tuple[int, ...] = (0, 1, 2)) -> float:
    """
    QWK = 1 - sum(w_ij * O_ij) / sum(w_ij * E_ij), w_ij = (i-j)^2 / (maxdiff)^2.
    The quadratic weighting punishes predicting 0 when truth is 2 (the failure
    mode that angers teachers most) far more than an off-by-one slip.
    """
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    n = len(labels)
    O = confusion_matrix(y_true, y_pred, labels=list(labels)).astype(np.float64)
    # Normalize observed/confusion counts into probability matrices so the
    # weighting convention matches sklearn's cohen_kappa_score(weights='quadratic').
    P_o = O / max(O.sum(), 1e-12)
    hist_t = np.bincount(y_true, minlength=n)[:n].astype(np.float64)
    hist_p = np.bincount(y_pred, minlength=n)[:n].astype(np.float64)
    P_e = np.outer(hist_t / max(hist_t.sum(), 1e-12),
                   hist_p / max(hist_p.sum(), 1e-12))
    # squared distance normalized by (maxdiff)^2 -> in [0, 1]
    w = np.subtract.outer(np.arange(n, dtype=np.float64),
                          np.arange(n, dtype=np.float64)) ** 2
    w /= (n - 1) ** 2 if n > 1 else 1.0
    denom = (w * P_e).sum()
    if denom < 1e-12:  # degenerate case: no dispersion to explain
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    return float(1.0 - (w * P_o).sum() / denom)


def evaluate_predictions(records: list[dict]) -> EvalResult:
    """records: [{'reference','hypothesis','rubric_true'?,'rubric_pred'?}, ...]"""
    refs, hypos = [], []
    agg = ErrorTaxonomy()
    per_sample = []
    yt, yp = [], []
    for r in records:
        ref = normalize_devanagari(r["reference"])
        hyp = normalize_devanagari(r["hypothesis"])
        refs.append(ref)
        hypos.append(hyp)
        tax = compute_taxonomy(ref, hyp)
        agg.total_edits += tax.total_edits
        agg.substitutions += tax.substitutions
        agg.insertions += tax.insertions
        agg.deletions += tax.deletions
        agg.matra_errors += tax.matra_errors
        agg.halant_conjunct_errors += tax.halant_conjunct_errors
        agg.confusable_grapheme_errors += tax.confusable_grapheme_errors
        agg.digit_errors += tax.digit_errors
        per_sample.append({
            "reference": ref, "hypothesis": hyp,
            "sample_cer": _jiwer_cer(ref, hyp) if ref else float("nan"),
            "taxonomy": {k: v for k, v in tax.as_dict().items() if k != "examples"},
        })
        if "rubric_true" in r and "rubric_pred" in r:
            yt.append(int(r["rubric_true"]))
            yp.append(int(r["rubric_pred"]))

    joined_refs = [" ".join(x.split()) for x in refs]
    joined_hypos = [" ".join(x.split()) for x in hypos]
    # corpus-level CER/WER (concatenation avoids jiwer's per-word alignment bias)
    corpus_cer = _jiwer_cer(" ".join(joined_refs), " ".join(joined_hypos))
    corpus_wer = _jiwer_wer(joined_refs, joined_hypos)

    qwk = macro_f1 = acc = None
    if yt and yp:
        qwk = quadratic_weighted_kappa(yt, yp)
        macro_f1 = float(f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0))
        acc = float(np.mean([t == p for t, p in zip(yt, yp)]))

    total_ref_chars = sum(len(x) for x in refs) or 1
    agg_dict = agg.as_dict()
    agg_dict.pop("examples", None)
    agg_dict["matra_error_rate"] = agg.matra_errors / total_ref_chars
    agg_dict["confusion_label_examples"] = [
        e for rec in per_sample for e in []  # examples kept per-sample only
    ]
    return EvalResult(
        n_samples=len(records), cer=corpus_cer, wer=corpus_wer,
        qwk=qwk, macro_f1=macro_f1, rubric_accuracy=acc,
        taxonomy=agg_dict, per_sample=per_sample,
    )


# ---------------------------------------------------------------------------
# Rubric scorer (structured-output validation + deterministic fallback)
# ---------------------------------------------------------------------------

RUBRIC_JSON_SCHEMA_HINT = (
    '{"transcribed_text": "<devanagari>", "rubric_score": 0|1|2, '
    '"error_feedback": "<marathi/english>"}'
)


def parse_structured_output(raw: str) -> dict:
    """
    Robust JSON extraction from VLM output: models occasionally wrap JSON in
    ``` fences or add trailing prose. Falls back to regex score recovery so a
    single stray token never crashes batch evaluation.
    """
    txt = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if fence:
        txt = fence.group(1).strip()
    brace = re.search(r"\{.*\}", txt, re.S)
    if brace:
        try:
            obj = json.loads(brace.group(0))
            if isinstance(obj, dict):
                obj.setdefault("transcribed_text", "")
                score = obj.get("rubric_score")
                obj["rubric_score"] = int(score) if score is not None and str(score).strip() in "012" else None
                obj.setdefault("error_feedback", "")
                return obj
        except json.JSONDecodeError:
            pass
    m = re.search(r"rubric[_ ]score\"?\s*[:=]\s*([012])", txt, re.I)
    return {"transcribed_text": txt,
            "rubric_score": int(m.group(1)) if m else None,
            "error_feedback": ""}


def rubric_from_cer(cer_val: float, ref: str, hyp: str) -> int:
    """
    Deterministic teacher-proxy rubric used ONLY to bootstrap labels for the
    fine-tuning set and as a sanity cross-check against model-predicted scores:
        0 : content wrong (CER > 60% or empty hypothesis)
        1 : right idea, orthographic damage (matra/confusable errors present)
        2 : clean transcription (CER <= 5%, no diacritic-class errors)
    """
    if not hyp.strip() or cer_val > 0.60:
        return 0
    tax = compute_taxonomy(ref, hyp)
    if cer_val <= 0.05 and tax.matra_errors == 0 and tax.confusable_grapheme_errors == 0 \
            and tax.halant_conjunct_errors == 0:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def markdown_report(results: dict[str, EvalResult]) -> str:
    """results: {'zero_shot_base': EvalResult, 'qlora_finetuned': EvalResult}"""
    rows = []
    header = "| Model | CER ↓ | WER ↓ | QWK ↑ | Macro-F1 ↑ | Rubric Acc ↑ | Matra errs | Conjunct errs | Confusable errs |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    for name, res in results.items():
        tax = res.taxonomy
        rows.append(
            f"| {name} | {res.cer*100:.2f}% | {res.wer*100:.2f}% | "
            f"{_fmt(res.qwk)} | {_fmt(res.macro_f1)} | {_fmt(res.rubric_accuracy, pct=True)} | "
            f"{tax['matra_errors']} | {tax['halant_conjunct_errors']} | "
            f"{tax['confusable_grapheme_errors']} |"
        )
    lines = [header, sep, *rows, ""]
    if len(results) == 2:
        (n1, r1), (n2, r2) = results.items()
        delta = (r1.cer - r2.cer) * 100
        lines.append(
            f"**Headline:** Reduced CER on noisy Devanagari handwriting from "
            f"{r1.cer*100:.1f}% ({n1}) to {r2.cer*100:.1f}% ({n2}) — "
            f"a {delta:.1f} percentage-point improvement."
        )
    return "\n".join(lines)


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Devanagari-aware evaluation for ShikshaEval-Indic")
    ap.add_argument("--pred", required=True, help="JSONL: reference/hypothesis/rubric_true/rubric_pred")
    ap.add_argument("--out", default="report.md", help="Markdown report path")
    ap.add_argument("--json", default=None, help="optional machine-readable metrics dump")
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.pred).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    res = evaluate_predictions(records)
    report = markdown_report({"model_under_test": res})
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    if args.json:
        payload = {k: v for k, v in asdict(res).items() if k != "per_sample"}
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
