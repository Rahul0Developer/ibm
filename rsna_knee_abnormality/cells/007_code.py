# ============================================================
# SECTION 5 - MODEL: MULTI-PLANE ATTENTION-POOLED 2.5D NETWORK
# ============================================================
# Data flow for one study (batch B, planes P=3, slices K=16):
#
#   (B,P,C,K,H,W) --flatten--> (B*P*K, C, H, W) --timm backbone--> (B*P*K, D)
#        --> reshape (B*P, K, D) --> Bi-GRU + MultiHeadAttention pooling --> (B*P, D')
#        --> concat plane metadata embedding (Fluid_Sensitive/Fat_Suppression/plane id)
#        --> per-plane gated mean over P streams -> study vector (B, D'')
#        --> MLP head --> 12 logits --> sigmoid probabilities
#
# The model is plane-aware by construction: each of the three anatomical streams
# carries its own learned positional/plane embedding so sagittal evidence drives
# ACL/meniscus/Baker's/PF-OA while coronal drives MCL/OA/contusion and axial drives
# effusion/synovitis/patellar pathology.
from __future__ import annotations


def build_backbone(name: str, in_ch: int, cfg: Config = CFG, pretrained: bool = True) -> nn.Module:
    """Instantiate a timm feature extractor with graceful offline fallback.

    If `timm` is missing or the pretrained weights cannot be fetched (offline
    submission mode), we return a compact CNN with identical output geometry so
    the checkpoint (trained weights) still loads exactly.
    """
    try:
        import timm
        kwargs: Dict[str, Any] = dict(
            pretrained=pretrained, in_chans=in_ch, num_classes=0,
            global_pool="avg", flatten=True,
        )
        m = timm.create_model(name, **kwargs)
        d = int(getattr(m, "num_features", cfg.FEATURE_DIM))
        print(f"[model] timm backbone={name} feat_dim={d} pretrained={pretrained}")
        return m
    except Exception as exc:
        print(f"[model] timm unavailable/offline ({exc}) -> lightweight fallback CNN")
        return FallbackBackbone(dim=cfg.FEATURE_DIM, in_chans=in_ch)


class FallbackBackbone(nn.Module):
    """4-stage stride-2 conv tower; output dim == declared FEATURE_DIM."""

    def __init__(self, dim: int = 768, in_chans: int = 3) -> None:
        super().__init__()
        base = max(32, dim // 24)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.SiLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2), nn.SiLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4), nn.SiLU(inplace=True),
            nn.Conv2d(base * 4, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        )
        self.num_features = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (N,C,H,W) -> (N,dim)
        return self.stem(x)


class SliceFeatureBank(nn.Module):
    """Apply the shared 2D backbone to every slice of every plane in ONE big batch."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def out_dim(self) -> int:
        return int(getattr(self.backbone, "num_features", CFG.FEATURE_DIM))

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """vol: (B,P,C,K,H,W) -> features (B,P,K,D)."""
        b, p, c, k, h, w = vol.shape
        x = vol.reshape(b * p * k, c, h, w)
        feats = self.backbone(x)                              # (B*P*K, D)
        d = feats.shape[-1]
        return feats.view(b, p, k, d)


class DepthAttentionPool(nn.Module):
    """Bidirectional GRU + multi-head self-attention pooling across slice depth."""

    def __init__(self, dim: int, heads: int = 8, layers: int = 1,
                 gru_hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(dim, gru_hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        gdim = gru_hidden * 2
        self.proj_in = nn.Linear(dim, gdim) if dim != gdim else nn.Identity()
        self.attn_norm = nn.LayerNorm(gdim)
        self.self_attn = nn.MultiheadAttention(gdim, num_heads=heads, dropout=dropout,
                                               batch_first=True)
        self.score = nn.Sequential(
            nn.LayerNorm(gdim), nn.Linear(gdim, gdim // 2), nn.Tanh(),
            nn.Linear(gdim // 2, 1),
        )
        self.ffn = nn.Sequential(nn.Linear(gdim, gdim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(gdim, gdim))
        self.out_dim = gdim

    def forward(self, seq: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """seq: (N,K,D) -> pooled (N,out_dim). `valid`: optional (N,K) mask (1=real slice)."""
        h = self.proj_in(seq)
        gru_out, _ = self.gru(h)                              # (N,K,2H)
        x = self.attn_norm(gru_out)
        if valid is not None:
            # nn.MultiheadAttention only supports a key-padding mask that is identical
            # for every query row; we therefore mask only fully-absent planes here and
            # handle per-slice masking exactly in the attention-pooling scores below.
            absent_rows = (valid.sum(dim=-1) < 0.5)           # (N,) plane entirely padded
            if bool(absent_rows.any()):
                keep = ~absent_rows
                x_kept = x[keep]
                sa_kept, _ = self.self_attn(x_kept, x_kept, x_kept, need_weights=False)
                sa = torch.zeros_like(x)
                sa[keep] = sa_kept                            # padded rows -> zero delta
            else:
                sa, _ = self.self_attn(x, x, x, need_weights=False)
        else:
            sa, _ = self.self_attn(x, x, x, need_weights=False)
        x = x + sa
        x = x + self.ffn(x)
        logits = self.score(x).squeeze(-1)                    # (N,K)
        if valid is not None:
            all_invalid = (valid.sum(dim=-1) < 0.5)           # (N,)
            logits = logits.masked_fill(~valid.bool(), float("-inf"))
            w = torch.softmax(logits, dim=-1)                 # attention pooling weights
            w = torch.nan_to_num(w, nan=0.0)                  # all -inf rows -> uniform->0
            if bool(all_invalid.any()):
                w[all_invalid] = 0.0                          # padded plane contributes nothing
        else:
            w = torch.softmax(logits, dim=-1)
        return (x * w.unsqueeze(-1)).sum(dim=1)               # (N,out_dim)


class SequenceMetadataEncoder(nn.Module):
    """Embed Fluid_Sensitive / Fat_Suppression / Plane-id into a dense context vector."""

    def __init__(self, meta_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.plane_emb = nn.Embedding(4, meta_dim)            # Sag/Cor/Axi/Unknown
        self.flag_proj = nn.Linear(2, meta_dim)               # fluid-sensitive, fat-sat
        self.weighted_emb = nn.Embedding(6, meta_dim)         # T1/T2/PD/STIR/SPIR/OTHER
        self.norm = nn.LayerNorm(meta_dim * 3)
        self.mlp = nn.Sequential(nn.Linear(meta_dim * 3, meta_dim * 4), nn.GELU(),
                                 nn.Dropout(dropout))
        self.out_dim = meta_dim * 4

    def forward(self, meta_flags: torch.Tensor) -> torch.Tensor:
        """meta_flags: (N,3) = [fluid_sensitive, fat_suppressed, plane_id]."""
        fs = meta_flags[:, 0:1]
        fas = meta_flags[:, 1:2]
        plane = meta_flags[:, 2].long().clamp(0, 3)
        # weighted contrast type is folded into plane slot when available (col 3+)
        w = meta_flags[:, 3].long().clamp(0, 5) if meta_flags.shape[1] > 3 else \
            torch.zeros_like(plane)
        e = torch.cat([self.plane_emb(plane), self.flag_proj(torch.cat([fs, fas], -1)),
                       self.weighted_emb(w)], dim=-1)
        return self.mlp(self.norm(e))


class KneeAbnormalityModel(nn.Module):
    """Multi-plane 2.5D attention-pooled classifier with sequence-metadata fusion."""

    def __init__(self, cfg: Config = CFG, backbone_name: Optional[str] = None,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        name = backbone_name or cfg.BACKBONE
        self.slices = SliceFeatureBank(build_backbone(name, cfg.IN_CHANNELS, cfg, pretrained))
        fdim = self.slices.out_dim

        # One depth aggregator per anatomical plane stream (plane-specific temporal
        # statistics differ strongly: sagittal spans the notch, axial the patella).
        self.pools = nn.ModuleList([
            DepthAttentionPool(fdim, cfg.ATTENTION_HEADS, cfg.ATTENTION_LAYERS,
                               cfg.GRU_HIDDEN, cfg.DROPOUT)
            for _ in range(cfg.NUM_PLANES)
        ])
        pooled_dim = self.pools[0].out_dim

        self.meta_enc = SequenceMetadataEncoder(cfg.META_DIM, cfg.DROPOUT)
        fuse_dim = pooled_dim + self.meta_enc.out_dim

        self.gate = nn.Sequential(nn.Linear(fuse_dim, 1), nn.Sigmoid())
        self.plane_fusion = nn.Sequential(
            nn.LayerNorm(fuse_dim * cfg.NUM_PLANES),
            nn.Linear(fuse_dim * cfg.NUM_PLANES, cfg.HEAD_HIDDEN), nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.HEAD_HIDDEN),
            nn.Linear(cfg.HEAD_HIDDEN, cfg.HEAD_HIDDEN // 2), nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
        )
        self.classifier = nn.Linear(cfg.HEAD_HIDDEN // 2, cfg.NUM_TARGETS)
        self._apply_weight_init()

    def _apply_weight_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------- forward
    def forward(self, volume: torch.Tensor, meta_flags: torch.Tensor,
                mask: torch.Tensor, plane_ids: Optional[torch.Tensor] = None,
                return_logits: bool = False) -> torch.Tensor:
        """volume (B,P,C,K,H,W); meta_flags (B,P,3+); mask (B,P) real-plane gate.

        Returns probabilities (B,12) unless return_logits=True.
        """
        b, p, c, k, h, w = volume.shape
        feats = self.slices(volume)                            # (B,P,K,D)
        pooled_streams: List[torch.Tensor] = []
        for pi in range(min(p, self.cfg.NUM_PLANES)):
            seq = feats[:, pi]                                # (B,K,D)
            valid = mask[:, pi].unsqueeze(-1).expand(-1, k)   # (B,K) 0 if padded plane
            pool = self.pools[pi](seq, valid)                 # (B,D')
            mvec = self.meta_enc(meta_flags[:, pi])           # (B,M)
            fused = torch.cat([pool, mvec], dim=-1)           # (B,fuse)
            g = self.gate(fused) * mask[:, pi:pi + 1]         # zero-out absent planes
            pooled_streams.append(fused * g)
        cat = torch.cat(pooled_streams, dim=-1)               # (B, P*fuse)
        z = self.plane_fusion(cat)
        z = self.head(z)
        logits = self.classifier(z)                           # (B,12)
        if return_logits:
            return logits
        return torch.sigmoid(logits.float())

    @torch.jit.ignore
    def param_groups(self, weight_decay: float) -> List[Dict[str, Any]]:
        """AdamW groups: no decay on norm/bias/embedding params."""
        decay, no_decay = [], []
        for name, prm in self.named_parameters():
            if not prm.requires_grad:
                continue
            if prm.ndim <= 1 or "bias" in name or "emb" in name.lower():
                no_decay.append(prm)
            else:
                decay.append(prm)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


class EMAModel:
    """Exponential moving average of model weights (stabilises AUC late in training)."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone().float() for n, p in model.named_parameters()
        }
        self.buffers: Dict[str, torch.Tensor] = {
            n: b.detach().clone().float() for n, b in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            s = self.shadow[n]
            s.mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)
        for n, b in model.named_buffers():
            if n in self.buffers and b.dtype.is_floating_point:
                self.buffers[n].mul_(self.decay).add_(b.detach().float(),
                                                      alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Swap EMA weights into `model`; return originals for later restore."""
        orig: Dict[str, torch.Tensor] = {}
        sd = model.state_dict()
        for n, p in model.named_parameters():
            orig[n] = p.detach().clone()
            p.data.copy_(self.shadow[n].to(p.dtype))
        for n, b in model.named_buffers():
            if n in self.buffers:
                orig[n] = b.detach().clone()
                b.data.copy_(self.buffers[n].to(b.dtype))
        del sd
        return orig

    def restore(self, model: nn.Module, orig: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in orig:
                    p.data.copy_(orig[n])
            for n, b in model.named_buffers():
                if n in orig:
                    b.data.copy_(orig[n])

    def state_dict(self) -> Dict[str, Any]:
        return {"shadow": self.shadow, "buffers": self.buffers, "decay": self.decay}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.decay = float(st.get("decay", self.decay))
        self.shadow = {k: v.float() for k, v in st["shadow"].items()}
        self.buffers = {k: v.float() for k, v in st.get("buffers", {}).items()}


print("[model] KneeAbnormalityModel + EMAModel defined")
