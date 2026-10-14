"""
MODULE 2 — Clinical Report Weak-Supervision Pseudo-Labeler.

Only a fraction of the RSNA training studies carry human-verified ground-truth labels,
but ALL training rows contain a free-text `Report`. This module implements a
deterministic, clinically-grounded rule-based NLP pipeline (regex + NegEx-style
negation scoping) to extract pseudo-labels for the 12 target abnormalities:

    ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
    Effusion, Synovitis, Baker's, Contusion, Fracture

Outputs:
    train_pseudo_labels.csv  — expanded label file:
        * ground-truth rows preserved verbatim (label_source == "gt")
        * unlabeled rows filled with report-derived labels (label_source == "pseudo")
    pseudo_label_validation.txt / .json — F1 & accuracy vs the labeled subset.

Kaggle usage:
    !python parse_reports.py --train-csv /kaggle/input/.../train.csv --out-csv train_pseudo_labels.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TARGET_COLS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


# --------------------------------------------------------------------------------------
# Condition lexicons: (target -> list of positive mention regexes)
# --------------------------------------------------------------------------------------
def _rx(*patterns: str) -> re.Pattern:
    """Compile alternation of lowercase patterns with word boundaries where sensible."""
    joined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(joined, flags=re.IGNORECASE | re.DOTALL)


CONDITION_PATTERNS: Dict[str, re.Pattern] = {
    # Ligaments
    "ACL": _rx(
        r"\bacl\b[^.;]{0,80}?(tear|disrupt|ruptur|sprain|avuls|mucoid|attenuat|insufficient|non[- ]visualiz)",
        r"(anterior cruciate ligament)[^.;]{0,80}?(tear|disrupt|ruptur|sprain|avuls|attenuat)",
        r"(acl graft)[^.;]{0,60}?(fail|loose|torn|ruptur)",
        r"(complete|partial|full[- ]thickness|high[- ]grade|low[- ]grade|grade [iI]{1,3})[^.;]{0,40}?\bacl\b",
        r"\bacl\b[^.;]{0,40}?(complete|partial|full[- ]thickness) tear",
    ),
    "MCL": _rx(
        r"\bmcl\b[^.;]{0,80}?(tear|sprain|ruptur|disrupt|edema along|thickening|increased signal)",
        r"(medial collateral ligament)[^.;]{0,80}?(tear|sprain|ruptur|disrupt|edema)",
        r"(edema along the (?:medial )?ligament)",
        r"\bmcl\b[^.;]{0,40}?grade [iI]{1,3}",
    ),
    # Menisci — require laterality adjacency to avoid cross-contamination.
    "Medial Meniscus": _rx(
        r"medial menisc[^.;]{0,80}?(tear|macera|complex tear|horizontal tear|radial tear|root tear|bucket|parameniscal|displac|degenerat|truncat)",
        r"(tear|macera|complex tear|horizontal tear|radial tear|root tear|bucket[- ]handle|parameniscal|degenerat\w* cyst|truncat\w*)[^.;]{0,40}?medial menisc",
        r"(medial)[^.;]{0,20}?(meniscal tear|meniscus tear)",
    ),
    "Lateral Meniscus": _rx(
        r"lateral menisc[^.;]{0,80}?(tear|macera|complex tear|horizontal tear|radial tear|root tear|bucket|parameniscal|displac|degenerat|truncat)",
        r"(tear|macera|complex tear|horizontal tear|radial tear|root tear|bucket[- ]handle|parameniscal|degenerat\w* cyst|truncat\w*)[^.;]{0,40}?lateral menisc",
        r"(lateral)[^.;]{0,20}?(meniscal tear|meniscus tear)",
    ),
    # Osteoarthritis compartments
    "Medial OA": _rx(
        r"(medial compartment\w*)[^.;]{0,60}?(narrow|joint space loss|osteoarthr|arthritis|cartilage loss|bone marrow lesion|subchondral)",
        r"(medial tibiofemoral)[^.;]{0,60}?(osteoarthr|arthritis|narrowing|cartilage loss)",
        r"(osteophyte[s]?)[^.;]{0,50}?medial",
        r"medial[^.;]{0,30}?(osteophyte[s]?|subchondral cyst|sclerosis)",
        r"(moderate|severe|mild)[^.;]{0,30}?(medial compartment|medial tibiofemoral) osteoarthr",
    ),
    "Lateral OA": _rx(
        r"(lateral compartment\w*)[^.;]{0,60}?(narrow|joint space loss|osteoarthr|arthritis|cartilage loss|bone marrow lesion|subchondral)",
        r"(lateral tibiofemoral)[^.;]{0,60}?(osteoarthr|arthritis|narrowing|cartilage loss)",
        r"(osteophyte[s]?)[^.;]{0,50}?lateral",
        r"lateral[^.;]{0,30}?(osteophyte[s]?|subchondral cyst|sclerosis)",
        r"(moderate|severe|mild)[^.;]{0,30}?(lateral compartment|lateral tibiofemoral) osteoarthr",
    ),
    "PF OA": _rx(
        r"(patellofemoral)[^.;]{0,60}?(osteoarthr|arthritis|arthrosis|cartilage loss|narrowing|spurr|osteophyte)",
        r"(trochlear groove)[^.;]{0,60}?(arthros|osteoarthr|cartilage loss|wear)",
        r"(patellar cartilage loss|patellar facet[^.;]{0,40}?(osteoarthr|cartilage loss))",
        r"(retropatellar)[^.;]{0,60}?(arthros|osteoarthr|cartilage loss|chondral)",
    ),
    # Fluid / synovium
    "Effusion": _rx(
        r"(joint effusion|suprapatellar effusion|knee joint effusion|large effusion|moderate effusion|small effusion|"
        r"joint fluid|fluid distension of the (?:joint|suprapatellar bursa)|effusion)",
    ),
    "Synovitis": _rx(
        r"(synovitis|synovial thickening|synovial proliferation|synovial enhancement|hypertrophic synovium|villonodular)",
    ),
    "Baker's": _rx(
        r"(baker'?s? cyst|popliteal cyst|semimembranosus[- ]gastrocnemius bursitis|"
        r"gastrocnemius[- ]semimembranosus bursitis|cyst in the popliteal fossa)",
    ),
    # Bone
    "Contusion": _rx(
        r"(bone contusion|bone bruise|bone marrow edema|bone marrow oedema|subchondral edema|subchondral fracture[^.;]{0,40}?(edema)?|"
        r"marrow edema pattern|post[- ]traumatic edema[^.;]{0,40}(femur|tibia))",
        r"(femoral|tibial|patellar)[^.;]{0,30}?(contusion|bone bruise|marrow edema)",
    ),
    "Fracture": _rx(
        r"(fractur\w*|cortical breach|cortical step[- ]?off|avulsion(?: fracture)?|displaced fragment|break in the cortex)",
        r"(tibial plateau|femoral condyle|fibular head|patella)[^.;]{0,40}?(break|depress\w*|comminut\w*|non[- ]?displaced fracture)",
    ),
}

NEGATION_CUES: List[str] = [
    "no evidence of", "no evidence for", "without", "free of", "no sign of", "no signs of",
    "intact", "unremarkable", "normal", "negative for", "not present", "no acute",
    "no demonstrative", "no", "none", "denies", "clear", "preserved", "grossly intact",
    "no significant", "without evidence", "rules out", "excluded",
]

# Positive-triggers that *cancel* a preceding negation within scope (NegEx "trigger").
CANCELLED_CUES: List[str] = [
    "but", "however", "although", "though", "still", "nevertheless", "except", "cannot exclude",
    "may represent", "possible", "suspected", "question of",
]

# Precomputed compiled cue sets
_NEG_RX = re.compile(r"\b(" + "|".join(re.escape(c) for c in NEGATION_CUES) + r")\b", re.I)
_CANCEL_RX = re.compile(r"\b(" + "|".join(re.escape(c) for c in CANCELLED_CUES) + r")\b", re.I)

# Scope windows (in characters) searched before the matched condition mention for neg cues.
NEG_SCOPE_BEFORE = 60
NEG_SCOPE_AFTER = 25


@dataclass
class PseudoLabelResult:
    labels: Dict[str, int]          # target -> 0/1
    confidence: Dict[str, float]    # target -> confidence in [0,1]
    mentions: Dict[str, List[str]]  # target -> matched snippets (audit trail)


def _is_negated(sentence: str, start: int, end: int) -> bool:
    """NegEx-lite: look for a negation cue immediately before the mention, and make sure
    no cancellation cue appears between the cue and the mention."""
    pre = sentence[max(0, start - NEG_SCOPE_BEFORE):start]
    post = sentence[end:end + NEG_SCOPE_AFTER]
    m = None
    for m in _NEG_RX.finditer(pre):
        pass  # take the LAST (closest) neg cue before the mention
    if m is None:
        # some cues may follow ("... is intact")
        pm = _NEG_RX.search(post)
        if pm is not None and not _CANCEL_RX.search(post[:pm.start()]):
            return True
        return False
    between = pre[m.end():]
    if _CANCEL_RX.search(between):
        return False
    # A sentence boundary (period) between cue and mention breaks negation scope.
    if "." in between and len(between.strip()) > 0 and between.strip().startswith("."):
        return False
    return True


def extract_pseudo_labels_from_report(report_text: str) -> PseudoLabelResult:
    """Rule-based multi-label extraction from one radiology report.

    Args:
        report_text: Free-text MRI knee radiology report. May be empty/NaN.
    Returns:
        PseudoLabelResult with binary labels, per-target confidence, audit snippets.
    """
    labels: Dict[str, int] = {t: 0 for t in TARGET_COLS}
    conf: Dict[str, float] = {t: 0.0 for t in TARGET_COLS}
    mentions: Dict[str, List[str]] = {t: [] for t in TARGET_COLS}

    if report_text is None or (isinstance(report_text, float) and np.isnan(report_text)):
        return PseudoLabelResult(labels, conf, mentions)
    text = str(report_text)
    if not text.strip():
        return PseudoLabelResult(labels, conf, mentions)

    # Split into sentences for local negation scoping while keeping offsets.
    sentences: List[Tuple[int, str]] = []
    pos = 0
    for chunk in re.split(r"(?<=[.;])\s+", text):
        idx = text.find(chunk, pos)
        if idx < 0:
            idx = pos
        sentences.append((idx, chunk))
        pos = idx + len(chunk)

    for target, rx in CONDITION_PATTERNS.items():
        n_pos = 0
        n_neg = 0
        for sent_start, sent in sentences:
            for m in rx.finditer(sent):
                snippet = m.group(0)[:120].strip()
                if _is_negated(sent, m.start(), m.end()):
                    n_neg += 1
                else:
                    n_pos += 1
                    if len(mentions[target]) < 3:
                        mentions[target].append(snippet)
        if n_pos > 0:
            labels[target] = 1
            # Confidence grows with number of independent positive mentions; negations reduce it.
            base = min(1.0, 0.7 + 0.1 * (n_pos - 1))
            conf[target] = round(base * (1.0 - 0.25 * min(n_neg, 2)), 3)
        else:
            labels[target] = 0
            # High-confidence negatives only when the anatomy was discussed but negated.
            discussed = any(rx.search(s) for _, s in sentences)
            conf[target] = round(0.85 if (discussed and n_neg > 0) else 0.6, 3)
    return PseudoLabelResult(labels, conf, mentions)


# --------------------------------------------------------------------------------------
# Validation against ground truth
# --------------------------------------------------------------------------------------
def validate_pseudo_labels(gt_df: pd.DataFrame, reports: pd.Series, threshold: float = 0.6) -> Dict[str, object]:
    """Compare extracted pseudo-labels against available ground-truth rows.

    Args:
        gt_df: DataFrame containing StudyInstanceUID + the 12 TARGET_COLS (0/1).
        reports: Series indexed identically to gt_df with the report text.
        threshold: minimum confidence required to accept a pseudo-label as "positive".
    Returns:
        dict with per-target and macro precision/recall/F1/accuracy.
    """
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    y_true_all: List[List[int]] = []
    y_pred_all: List[List[int]] = []
    per_target_stats: Dict[str, Dict[str, float]] = {}

    aligned = gt_df.reset_index(drop=True)
    rep = reports.reindex(aligned.index).reset_index(drop=True) if hasattr(reports, "reindex") else pd.Series(list(reports))

    for i in range(len(aligned)):
        res = extract_pseudo_labels_from_report(rep.iloc[i])
        yt = [int(aligned.iloc[i][t]) for t in TARGET_COLS]
        yp = [1 if (res.labels[t] == 1 and res.confidence[t] >= threshold) else 0 for t in TARGET_COLS]
        y_true_all.append(yt)
        y_pred_all.append(yp)

    y_true_np = np.array(y_true_all, dtype=np.int8)
    y_pred_np = np.array(y_pred_all, dtype=np.int8)

    for j, t in enumerate(TARGET_COLS):
        try:
            f1_t = float(f1_score(y_true_np[:, j], y_pred_np[:, j], zero_division=0))
        except Exception:
            f1_t = float("nan")
        acc_t = float(accuracy_score(y_true_np[:, j], y_pred_np[:, j]))
        per_target_stats[t] = {"f1": f1_t, "accuracy": acc_t,
                               "prevalence_gt": float(y_true_np[:, j].mean()),
                               "prevalence_pred": float(y_pred_np[:, j].mean())}

    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_np, y_pred_np, average="micro", zero_division=0))
    exact_acc = float(np.mean(np.all(y_true_np == y_pred_np, axis=1)))
    sample_hamming = float(np.mean(np.sum(y_true_np != y_pred_np, axis=1)) / len(TARGET_COLS))

    # AUC proxy using confidence-weighted positives where computable.
    auc_scores: List[float] = []
    for j, t in enumerate(TARGET_COLS):
        scores = np.array([extract_pseudo_labels_from_report(rep.iloc[i]).confidence[t] *
                           extract_pseudo_labels_from_report(rep.iloc[i]).labels[t]
                           for i in range(min(len(rep), 300))])
        try:
            if len(np.unique(y_true_np[:len(scores), j])) > 1:
                auc_scores.append(float(roc_auc_score(y_true_np[:len(scores), j], scores)))
        except Exception:
            pass
    mean_auc_proxy = float(np.mean(auc_scores)) if auc_scores else float("nan")

    summary = {
        "n_validated": int(len(y_true_np)),
        "per_target": per_target_stats,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "exact_match_accuracy": exact_acc,
        "hamming_loss_per_label": sample_hamming,
        "mean_auc_confidence_proxy": mean_auc_proxy,
        "threshold_used": threshold,
    }
    return summary


# --------------------------------------------------------------------------------------
# Expansion pipeline: GT + pseudo-labels
# --------------------------------------------------------------------------------------
def build_expanded_labels(
    train_csv_path: str | Path,
    out_csv_path: str | Path,
    validation_out_prefix: Optional[str | Path] = None,
    confidence_threshold: float = 0.6,
    max_pseudo_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Read train.csv, keep ground-truth rows untouched, fill unlabeled rows with report-derived labels.

    The official RSNA release marks unlabeled rows by NaN in the target columns while still
    providing the Report text. Rows whose entire label vector is NaN become pseudo-labeled.
    """
    df = pd.read_csv(train_csv_path)
    if "StudyInstanceUID" not in df.columns:
        raise ValueError("train.csv must contain StudyInstanceUID column")
    has_report = "Report" in df.columns
    if not has_report:
        raise ValueError("train.csv must contain the free-text 'Report' column for weak supervision")

    labeled_mask = df[TARGET_COLS].notna().any(axis=1)
    gt_df = df.loc[labeled_mask, ["StudyInstanceUID"] + TARGET_COLS].copy()
    unl_df = df.loc[~labeled_mask, ["StudyInstanceUID", "Report"]].copy()

    logger_lines: List[str] = []
    logger_lines.append(f"Ground-truth labeled studies : {len(gt_df)}")
    logger_lines.append(f"Unlabeled studies w/ reports : {len(unl_df)}")

    # --- Validate parser on the labeled subset first -------------------------------
    val_summary: Dict[str, object] = {}
    if len(gt_df) > 0:
        reports_for_gt = df.loc[labeled_mask, "Report"].reset_index(drop=True)
        try:
            val_summary = validate_pseudo_labels(gt_df, reports_for_gt, threshold=confidence_threshold)
            logger_lines.append(
                f"Parser validation on GT subset: macro-F1={val_summary.get('macro_f1', float('nan')):.3f}, "
                f"exact-match-acc={val_summary.get('exact_match_accuracy', float('nan')):.3f}"
            )
        except ImportError:
            logger_lines.append("sklearn unavailable in this environment; skipped validation metrics.")

    # --- Generate pseudo labels -----------------------------------------------------
    if max_pseudo_rows is not None:
        unl_df = unl_df.head(max_pseudo_rows)

    rows: List[Dict[str, object]] = []
    for uid, rpt in zip(unl_df["StudyInstanceUID"].values, unl_df["Report"].values):
        res = extract_pseudo_labels_from_report("" if pd.isna(rpt) else str(rpt))
        row: Dict[str, object] = {"StudyInstanceUID": uid, "label_source": "pseudo"}
        for t in TARGET_COLS:
            lab = res.labels[t]
            cnf = res.confidence[t]
            # Only trust positives above threshold; keep confident negatives as 0.
            final_lab = 1 if (lab == 1 and cnf >= confidence_threshold) else 0
            row[t] = final_lab
        row["pseudo_mean_conf"] = float(np.mean(list(res.confidence.values())))
        rows.append(row)
    pseudo_df = pd.DataFrame(rows, columns=["StudyInstanceUID", "label_source"] + TARGET_COLS + ["pseudo_mean_conf"])

    gt_out = gt_df.copy()
    gt_out["label_source"] = "gt"
    combined = pd.concat([gt_out, pseudo_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="StudyInstanceUID", keep="first")

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv_path, index=False)
    logger_lines.append(f"Wrote expanded labels: {out_csv_path} ({len(combined)} rows)")

    if validation_out_prefix is not None:
        prefix = Path(validation_out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with open(prefix.with_suffix(".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(logger_lines) + "\n\n")
            fh.write(json.dumps(val_summary, indent=2, default=float))
        with open(prefix.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump({"log": logger_lines, "validation": val_summary}, fh, indent=2, default=float)

    print("\n".join(logger_lines))
    return combined


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RSNA Knee report weak-supervision pseudo-labeler")
    ap.add_argument("--train-csv", required=True, help="Path to train.csv (with Report column)")
    ap.add_argument("--out-csv", required=True, help="Output expanded labels csv")
    ap.add_argument("--validation-prefix", default=None, help="Write validation report txt/json here")
    ap.add_argument("--confidence-threshold", type=float, default=0.6)
    ap.add_argument("--max-pseudo-rows", type=int, default=None)
    ns = ap.parse_args(argv)

    build_expanded_labels(
        train_csv_path=ns.train_csv,
        out_csv_path=ns.out_csv,
        validation_out_prefix=ns.validation_prefix,
        confidence_threshold=ns.confidence_threshold,
        max_pseudo_rows=ns.max_pseudo_rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
