# ============================================================
# SECTION 6 - LOSS FUNCTIONS & COMPETITION METRIC
# ============================================================
from __future__ import annotations


class AsymmetricLossOptimized(nn.Module):
    """Asymmetric Loss (Ridnik et al., 2021) tuned for extreme multi-label sparsity.

    prob_true = p, prob_false = 1-p
      positive: max(1 - p - m_p, 0) ** g_p * log(p)
      negative: max(p - m_n, 0) ** g_n * log(1 - p)     with p shifted by clip -> p'=min(p,1-clip)

    gamma_neg >> gamma_pos de-emphasises easy negatives while preserving every hard
    positive; the probability *shift* (clip) additionally saturates very confident
    negatives to zero gradient. This maximises AUC under ~5-15% positive rates.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05,
                 margin_neg: float = 0.0, margin_pos: float = 0.0, disable_torch_grad_focalize: bool = False,
                 eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip
        self.margin_neg, self.margin_pos = margin_neg, margin_pos
        self.disable = disable_torch_grad_focalize
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """logits: raw pre-sigmoid (B,T); targets in {0,1}; sample_weight (B,) or (B,T)."""
        x_sigmoid = torch.sigmoid(logits)
        xs_pos, xs_neg = x_sigmoid, 1.0 - x_sigmoid
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)       # probability shifting

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable:
                x_sigmoid = x_sigmoid.detach()
                xs_pos, xs_neg = x_sigmoid, 1.0 - x_sigmoid
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)                       # pt = p if t>0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss *= one_sided_w

        if self.margin_pos > 0 or self.margin_neg > 0:
            dm = targets * self.margin_pos + (1 - targets) * self.margin_neg
            loss *= torch.relu((1.0 - torch.sigmoid(logits) * (2 * targets - 1)) - dm + 1.0).clamp(max=1.0)

        if sample_weight is not None:
            w = sample_weight
            if w.dim() == 1:
                w = w.unsqueeze(-1)
            loss = loss * w
            return -loss.sum() / w.expand_as(loss).sum().clamp(min=1.0)
        return -loss.mean()


class FocalBCEWithLogits(nn.Module):
    """Alternative focal-BCE objective (kept selectable via CFG for ablations)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = at * (1 - pt) ** self.gamma * bce
        if sample_weight is not None:
            w = sample_weight.unsqueeze(-1) if sample_weight.dim() == 1 else sample_weight
            return (loss * w).sum() / w.expand_as(loss).sum().clamp(min=1.0)
        return loss.mean()


def make_loss(cfg: Config = CFG) -> nn.Module:
    return AsymmetricLossOptimized(
        gamma_neg=cfg.ASL_GAMMA_NEG, gamma_pos=cfg.ASL_GAMMA_POS, clip=cfg.ASL_CLIP,
        margin_neg=cfg.ASL_MARGIN_NEG, margin_pos=cfg.ASL_MARGIN_POS,
    )


# ------------------------------------------------------------------- metric
def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC that never crashes when a batch/label contains a single class."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0:
        return float("nan")
    uniq = np.unique(y_true)
    if len(uniq) < 2:                    # degenerate -> undefined AUC, return neutral 0.5
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def macro_average_roc_auc(labels: np.ndarray, preds: np.ndarray,
                          target_cols: Sequence[str] = TARGET_COLS) -> Tuple[float, Dict[str, float]]:
    """Competition score: mean of per-target ROC-AUC over the 12 abnormalities."""
    labels = np.asarray(labels, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32)
    per: Dict[str, float] = {}
    for i, col in enumerate(target_cols):
        per[col] = safe_roc_auc(labels[:, i], preds[:, i])
    vals = np.array([v for v in per.values() if not math.isnan(v)], dtype=np.float64)
    macro = float(vals.mean()) if vals.size else 0.5
    return macro, per


class MetricLogger:
    """Accumulates epoch predictions and reports macro-AUC + per-class breakdown."""

    def __init__(self, target_cols: Sequence[str] = TARGET_COLS) -> None:
        self.target_cols = list(target_cols)
        self.reset()

    def reset(self) -> None:
        self.uids: List[str] = []
        self.y: List[np.ndarray] = []
        self.p: List[np.ndarray] = []

    @torch.no_grad()
    def update(self, uids: Sequence[str], probs: torch.Tensor, labels: torch.Tensor) -> None:
        self.uids.extend(list(uids))
        self.y.append(labels.detach().float().cpu().numpy())
        self.p.append(probs.detach().float().cpu().numpy())

    def compute(self) -> Tuple[float, Dict[str, float], pd.DataFrame]:
        y = np.concatenate(self.y, 0) if self.y else np.zeros((0, len(self.target_cols)), np.float32)
        p = np.concatenate(self.p, 0) if self.p else np.zeros_like(y)
        macro, per = macro_average_roc_auc(y, p, self.target_cols)
        frame = pd.DataFrame({"StudyInstanceUID": self.uids, "y": list(y), "p": list(p)})
        return macro, per, frame

    @staticmethod
    def format(per: Dict[str, float], macro: float) -> str:
        rows = ", ".join(f"{k}={v:.4f}" for k, v in per.items())
        return f"[metric] MACRO-AUC={macro:.5f} | {rows}"


print("[loss] ASL(g_neg=4,g_pos=0,clip=0.05) + macro-averaged ROC-AUC logger ready")
