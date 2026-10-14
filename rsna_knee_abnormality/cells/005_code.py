# ============================================================
# SECTION 3 - WEAK SUPERVISION / RADIOLOGY REPORT PARSER
# ============================================================
# Only a subset of training studies carry human-verified labels, but EVERY study
# ships a free-text `Report`. This NLP mining stage converts reports into
# high-confidence pseudo-labels (expanding the effective train set 5x-10x).
#
# Design:
#   * sentence-level scoping (negation rarely crosses sentence boundaries)
#   * multi-plane negation cues with lookahead windows ("no evidence of X",
#     "X is intact", "unremarkable", "without", "negative for", "free of")
#   * uncertainty cues demote a mention to "soft" (excluded from hard labels)
#   * synonym lexicons per target + anatomy-aware context boost
#   * deterministic output: label vector + per-target confidence in [0,1]
from __future__ import annotations

import re
from enum import IntEnum


class MentionState(IntEnum):
    ABSENT = 0        # explicitly negated  -> hard negative (conf high)
    PRESENT = 1       # affirmative finding -> hard positive
    UNCERTAIN = 2     # equivocal/suspected -> soft, ignored by default
    UNMENTIONED = 3   # not discussed       -> soft negative (low conf)


# ----------------------------------------------------------- lexical resources
NEGATION_CUES: List[str] = [
    r"\bno\s+evidence(?:\s+of)?\b", r"\bno\s+(?:findings?|sign(?:s)?|data)\s+(?:to\s+)?(?:suggest|of)\b",
    r"\bnot\s+(?:seen|identified|detected|present|apparent)\b",
    r"\bwithout\s+(?:evidence\s+of|any|significant)?\b", r"\bnegative\s+for\b",
    r"\bfree\s+of\b", r"\bfreedom\s+from\b", r"\bno\b(?=[^.]{0,60}\b(?:tear|effusion|fracture|lesion|abnormality)\b)",
    r"\bnone\b", r"\bno\s+acute\b", r"\bunremarkable\b", r"\bnormal\b",
    r"\bintact\b", r"\bnormally\b", r"\bcannot\s+be\s+seen\b",
    r"\bnot\s+displaced\b", r"\bno\s+significant\b", r"\bclear\b",
    r"\bmaintained\b", r"\bpreserved\b", r"\bwithin\s+normal\s+limits\b",
    r"\bno\s+underlying\b", r"\bdenies\b", r"\babsence\s+of\b", r"\bwithout\b",
]
UNCERTAINTY_CUES: List[str] = [
    r"\bmay\b", r"\bmight\b", r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b",
    r"\bsuspect(?:ed)?\b", r"\bequivocal\b", r"\bcannot\s+exclude\b", r"\bconcern\s+for\b",
    r"\brule\s+out\b", r"\bquestion(?:able)?\b", r"\blimited?\b", r"\bpartial\b",
    r"\bversus\b", r"\bor\s+possibly\b", r"\bindeterminate\b", r"\bsuggests?\b",
    r"\blikely\b", r"\bquery\b", r"\bif\s+there\s+is\b", r"\bdifficult\s+to\s+assess\b",
]

NEG_RE = re.compile("|".join(NEGATION_CUES), flags=re.IGNORECASE)
UNC_RE = re.compile("|".join(UNCERTAINTY_CUES), flags=re.IGNORECASE)

# Synonym lexicon per competition target. Ordered most-specific first.
TARGET_LEXICON: Dict[str, List[str]] = {
    "ACL": [r"anterior\s+cruciate\s+(?:ligament)?", r"\bacl\b"],
    "MCL": [r"medial\s+collateral\s+(?:ligament)?", r"\bmcl\b",
            r"tibial\s+collateral\s+(?:ligament)?"],
    "Medial Meniscus": [r"medial\s+(?:menisc(?:us|al)|meniscal\s+(?:body|horn|posterior|anterior))",
                        r"\bmm\b\s+(?:menisc)", r"medial\s+menisci"],
    "Lateral Meniscus": [r"lateral\s+(?:menisc(?:us|al)|meniscal\s+(?:body|horn|posterior|anterior))",
                         r"\blm\b\s+(?:menisc)", r"lateral\s+menisci", r"popliteus\s+hiatus"],
    "Medial OA": [r"(?:medial\s+(?:tibiofemoral|compartment|tfj).*?(?:osteoarthritis|degenerat|arthritis|joint\s+space))",
                  r"(?:osteoarthritis|degenerative\s+change|joint\s+space\s+narrowing).*?medial\s+(?:tibiofemoral|compartment)",
                  r"medial\s+compartment\s+(?:osteoarthritis|oa|arthritis|degeneration)"],
    "Lateral OA": [r"(?:lateral\s+(?:tibiofemoral|compartment|tfj).*?(?:osteoarthritis|degenerat|arthritis|joint\s+space))",
                   r"(?:osteoarthritis|degenerative\s+change|joint\s+space\s+narrowing).*?lateral\s+(?:tibiofemoral|compartment)",
                   r"lateral\s+compartment\s+(?:osteoarthritis|oa|arthritis|degeneration)"],
    "PF OA": [r"(?:patellofemoral|pfj|p\.?f\.?\s+joint).*?(?:osteoarthritis|degenerat|arthritis|joint\s+space|spurring)",
              r"(?:osteoarthritis|degenerative\s+change).*?patellofemoral",
              r"patellofemoral\s+(?:compartment|joint)\s*(?:osteoarthritis|oa|arthritis)"],
    "Effusion": [r"(?:joint\s+)?effusion", r"joint\s+fluid", r"fluid\s+(?:in\s+the\s+)?joint",
                 r"\bhydrarthros(?:is|es)\b", r"suprapatellar\s+(?:fluid|effusion|pouch)",
                 r"intra[-\s]?articular\s+fluid"],
    "Synovitis": [r"synovitis", r"synovial\s+(?:thickening|proliferation|enhancement|hypertrophy|inflammation)",
                  r"villonodular", r"pigmented\s+villonodular"],
    "Baker's": [r"baker'?s?\s+cyst", r"popliteal\s+(?:burs(?:a|ae|al)|cyst)",
                r"gastrocnemius[-\s]semimembranosus\s+bursa"],
    "Contusion": [r"bone\s+(?:marrow\s+)?(?:edema|bruise|contusion)", r"(?:marrow|osseous)\s+(?:edema|bruising|contusion)",
                  r"subchondral\s+(?:edema|stress\s+reaction)", r"insufficiency\s+edema",
                  r"trabecular\s+(?:microfracture|edema)", r"\bcontusion\b"],
    "Fracture": [r"\bfractur(?:e|ed)\b", r"\bnon[-\s]?displaced\s+fracture\b", r"cortical\s+(?:break|step|irregularity)",
                 r"avulsion\s+(?:fracture|injury)", r"\bosteochondral\s+(?:fracture|lesion)\b", r"\bsubchondral\s+insufficiency\s+fracture\b"],
}

# Findings that are considered POSITIVE-by-construction when described as abnormal
# even without an explicit disease noun (used for OA proxies).
OA_PROXY_TERMS: Dict[str, List[str]] = {
    "Medial OA": [r"medial.*?(?:eburnation|osteophyte|subchondral\s+cyst)"],
    "Lateral OA": [r"lateral.*?(?:eburnation|osteophyte|subchondral\s+cyst)"],
    "PF OA": [r"patellofemoral.*?(?:osteophyte|subchondral\s+cyst|eburnation|cartilage\s+loss)"],
}

_COMPILED_TARGET: Dict[str, List[re.Pattern]] = {
    t: [re.compile(p, flags=re.IGNORECASE) for p in pats] for t, pats in TARGET_LEXICON.items()
}
_COMPILED_PROXY: Dict[str, List[re.Pattern]] = {
    t: [re.compile(p, flags=re.IGNORECASE) for p in pats] for t, pats in OA_PROXY_TERMS.items()
}

_SENT_SPLIT = re.compile(r"(?<=[.;])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]
    return parts if parts else ([text.strip()] if text and text.strip() else [])


def _classify_mention(sentence: str, match_start: int, match_end: int) -> MentionState:
    """Decide whether a matched finding inside `sentence` is negated / uncertain."""
    window_pre = sentence[max(0, match_start - 90):match_start]
    window_post = sentence[match_end:min(len(sentence), match_end + 90)]

    # A negation cue anywhere before the term within the clause scope dominates.
    if NEG_RE.search(window_pre):
        # Exception: "no effusion" style handled; but "intact ACL" -> negation AFTER term.
        return MentionState.ABSENT
    if NEG_RE.search(window_post):
        tail = window_post.lower()
        if any(k in tail for k in ("intact", "maintained", "preserved", "normal",
                                   "unremarkable", "no abnormal", "without")):
            return MentionState.ABSENT
    if UNC_RE.search(window_pre) or UNC_RE.search(window_post):
        return MentionState.UNCERTAIN
    return MentionState.PRESENT


def _scan_sentence(sentence: str) -> Dict[str, MentionState]:
    """Return best (most informative) state per target found in one sentence."""
    states: Dict[str, MentionState] = {}
    for target, pats in _COMPILED_TARGET.items():
        for pat in pats:
            m = pat.search(sentence)
            if m is None:
                continue
            st = _classify_mention(sentence, m.start(), m.end())
            prev = states.get(target)
            # priority: PRESENT > UNCERTAIN > ABSENT when multiple mentions exist
            order = {MentionState.PRESENT: 3, MentionState.UNCERTAIN: 2,
                     MentionState.ABSENT: 1, MentionState.UNMENTIONED: 0}
            if prev is None or order[st] >= order[prev]:
                states[target] = st
            break
        if target in states:
            continue
        for proxy in _COMPILED_PROXY.get(target, []):
            m = proxy.search(sentence)
            if m is not None:
                st = _classify_mention(sentence, m.start(), m.end())
                if st == MentionState.PRESENT:
                    states[target] = MentionState.PRESENT
                break
    return states


def extract_pseudo_labels_from_report(
    report_text: str,
    use_uncertain_as_negative: bool = True,
    min_positive_confidence: float = 0.60,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, MentionState]]:
    """Mine a 12-dim pseudo-label vector + per-target confidence from a report.

    Returns
    -------
    labels : np.ndarray (12,) float32 in {0,1}  -> argmax interpretation
    conf   : np.ndarray (12,) float32 in [0,1]  -> trust weight per target
    states : dict target -> MentionState (debug/inspection)
    """
    text = (report_text or "").replace("\n", " ")
    agg: Dict[str, MentionState] = {t: MentionState.UNMENTIONED for t in CFG.TARGET_COLS}
    votes: Dict[str, Dict[MentionState, int]] = {t: {s: 0 for s in MentionState} for t in CFG.TARGET_COLS}

    for sent in _split_sentences(text):
        st = _scan_sentence(sent)
        for target, state in st.items():
            votes[target][state] += 1

    for t in CFG.TARGET_COLS:
        v = votes[t]
        order = [MentionState.PRESENT, MentionState.ABSENT, MentionState.UNCERTAIN]
        best = max(order, key=lambda s: v[s])
        if v[best] == 0:
            agg[t] = MentionState.UNMENTIONED
        else:
            agg[t] = best

    labels = np.zeros(CFG.NUM_TARGETS, dtype=np.float32)
    conf = np.full(CFG.NUM_TARGETS, 0.20, dtype=np.float32)
    for i, t in enumerate(CFG.TARGET_COLS):
        st = agg[t]
        n_tot = sum(votes[t].values()) or 1
        if st == MentionState.PRESENT:
            labels[i] = 1.0
            conf[i] = min(1.0, 0.70 + 0.10 * votes[t][MentionState.PRESENT] / n_tot)
        elif st == MentionState.ABSENT:
            labels[i] = 0.0
            conf[i] = min(1.0, 0.80 + 0.05 * votes[t][MentionState.ABSENT] / n_tot)
        elif st == MentionState.UNCERTAIN:
            labels[i] = 0.0 if use_uncertain_as_negative else 0.5
            conf[i] = 0.35
        else:  # UNMENTIONED -> weak negative prior
            labels[i] = 0.0
            conf[i] = 0.20
    conf[labels == 1.0] = np.maximum(conf[labels == 1.0], min_positive_confidence)
    return labels, conf, agg


# ------------------------------------------------------- dataset assembly logic
def build_labelled_frame(
    train_df: pd.DataFrame,
    cfg: Config = CFG,
    report_col_candidates: Sequence[str] = ("Report", "report", "radiology_report", "impression"),
) -> pd.DataFrame:
    """Merge human ground-truth rows with report-mined pseudo-labelled rows.

    Guarantees:
      * GT rows keep their exact labels (label_source='gt', weight=1.0)
      * unlabelled rows get report-derived labels when available
      * adds columns: <target>_pseudo, pseudo_conf, label_weight, label_source
    """
    df = train_df.copy()
    rep_col = next((c for c in report_col_candidates if c in df.columns), None)

    gt_mask = df[TARGET_COLS].notna().all(axis=1) if all(c in df.columns for c in TARGET_COLS) \
        else pd.Series(False, index=df.index)

    pseudo = np.zeros((len(df), cfg.NUM_TARGETS), dtype=np.float32)
    conf = np.zeros((len(df), cfg.NUM_TARGETS), dtype=np.float32)
    source = np.array(["unknown"] * len(df), dtype=object)

    for pos, (_, row) in enumerate(df.iterrows()):
        uid_ok = bool(gt_mask.iloc[pos])
        if uid_ok:
            vec = row[TARGET_COLS].astype(np.float32).to_numpy()
            pseudo[pos] = vec
            conf[pos] = 1.0
            source[pos] = "gt"
            continue
        if rep_col is not None and isinstance(row.get(rep_col), str) and row[rep_col].strip():
            lab, cf, _ = extract_pseudo_labels_from_report(row[rep_col])
            pseudo[pos] = lab
            conf[pos] = cf
            source[pos] = "report"
        else:
            pseudo[pos] = 0.0
            conf[pos] = 0.0
            source[pos] = "none"

    for i, t in enumerate(TARGET_COLS):
        df[f"{t}_pseudo"] = pseudo[:, i]
    df["pseudo_conf"] = conf.mean(axis=1)
    df["label_source"] = source
    df["label_weight"] = np.where(
        source == "gt", 1.0,
        np.where(source == "report", conf.max(axis=1) * cfg.PSEUDO_LABEL_WEIGHT, 0.0)
    )
    # Rows with no signal at all must never enter the loss.
    df.loc[df["label_source"] == "none", "label_weight"] = 0.0
    return df


print("[reports] parser ready | targets=%d | negation_cues=%d | uncertainty_cues=%d"
      % (len(TARGET_LEXICON), len(NEGATION_CUES), len(UNCERTAINTY_CUES)))
