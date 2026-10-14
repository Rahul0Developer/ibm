# ============================================================
# SECTION 2 - ROBUST DICOM INGESTION & 3D VOLUME SAMPLING
# ============================================================
# Thread-safe, defensive series reader:
#   * fast header-only scan (deferred_read) -> metadata without pixel decode
#   * slice ordering by SliceLocation / ImagePositionPatient[2] / InstanceNumber
#   * anatomical-plane inference from the ImageOrientationPatient normal vector
#   * sequence flags: Fluid_Sensitive / Fat_Suppression / weighted contrast type
#   * pixel extraction with pydicom -> pylibjpeg -> gdcm -> imageio fallback chain
#   * percentile-clipped min-max normalisation + uniform K-slice linspace resample
# Output tensor geometry: (C=3, Depth=K, H, W).
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor


# --------------------------------------------------------------- plane helpers
PLANE_SAGITTAL, PLANE_CORONAL, PLANE_AXIAL, PLANE_UNKNOWN = 0, 1, 2, 3
PLANE_NAMES: List[str] = ["Sagittal", "Coronal", "Axial", "Unknown"]


def classify_orientation(iop: Optional[Sequence[float]]) -> int:
    """Map a 6-value ImageOrientationPatient to an anatomical plane index.

    The cross product of the row/column direction cosines gives the slice normal:
      normal ~ +Z -> Axial, ~ +Y -> Coronal, ~ +X -> Sagittal.
    """
    if iop is None or len(iop) != 6:
        return PLANE_UNKNOWN
    try:
        row = np.asarray(iop[:3], dtype=np.float64)
        col = np.asarray(iop[3:6], dtype=np.float64)
        nrm = np.cross(row, col)
        norm = np.linalg.norm(nrm)
        if norm < 1e-6:
            return PLANE_UNKNOWN
        nrm = nrm / norm
        abs_n = np.abs(nrm)
        axis = int(np.argmax(abs_n))          # 0=X(sag) 1=Y(cor) 2=Z(axi)
        if abs_n[0] > 0.75:
            return PLANE_SAGITTAL
        if abs_n[1] > 0.75:
            return PLANE_CORONAL
        if abs_n[2] > 0.75:
            return PLANE_AXIAL
        return PLANE_UNKNOWN                   # oblique -> treated as unknown
    except Exception:
        return PLANE_UNKNOWN


WEIGHTED_MAP: Dict[str, str] = {
    "PD": "PD", "PROTON DENSITY": "PD", "T1": "T1", "T2": "T2",
    "T2*": "T2_STAR", "T2 STAR": "T2_STAR", "STIR": "STIR", "SPIR": "SPIR",
    "DWI": "OTHER", "SWI": "T2_STAR",
}


def parse_sequence_metadata(desc: str, series_id: str, extra_tags: str = "") -> Dict[str, Any]:
    """Derive Fluid-Sensitivity / Fat-Suppression / contrast from free-text tags.

    RSNA knee MRI encodes these in SeriesDescription and (when present)
    ScanningSequence / SequenceVariant / InferenceType strings.
    """
    blob = f"{desc} {series_id} {extra_tags}".upper()
    words = set(blob.replace("/", " ").replace("-", " ").replace("_", " ").split())

    fluid_sensitive = bool(
        {"FLUID", "FLUIDSENSTIVE", "FLUIDSENSITIVE", "FS", "STIR", "PSIR", "SPAIR", "T2"} & words
        or "FLUID SENSITIVE" in blob or "FLUID-SENSITIVE" in blob
        or ("T2" in blob and ("SAT" in blob or "FS" in blob))
        or "STIR" in blob
    )
    fat_suppressed = bool(
        {"FATSAT", "FS", "SPAIR", "PSIR", "STIR", "SPIR", "FATSUPPRESSED"} & words
        or "FAT SAT" in blob or "FAT-SAT" in blob or "FAT SUPPRESSED" in blob
        or "FATSAT" in blob
    )
    weighted = "OTHER"
    for key, val in WEIGHTED_MAP.items():
        if key in words:
            weighted = val
            break
    return {
        "fluid_sensitive": int(fluid_sensitive),
        "fat_suppressed": int(fat_suppressed),
        "weighted": weighted,
        "description": desc,
    }


# ------------------------------------------------------------ pixel extraction
def _apply_rescale(ds: Any, arr: np.ndarray) -> np.ndarray:
    """Apply RescaleSlope/Intercept so intensities are comparable across scanners."""
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    out = arr.astype(np.float32) * slope + inter
    return np.nan_to_num(out, nan=0.0, posinf=float(arr.max(initial=0)), neginf=0.0)


def _decode_pixels_fallbacks(path: Path) -> Optional[np.ndarray]:
    """Unpack PixelData when pydicom's native handlers fail.

    Order: pydicom(+pylibjpeg) -> gdcm -> imageio -> raw numpy reinterpretation.
    Handles JPEG-Lossless / JPEG-2000 / RLE syntaxes found in RSNA MRI exports.
    """
    # ---- 1. pydicom with all registered decoders
    if HAVE_PYDICOM:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            arr = ds.pixel_array
            return _apply_rescale(ds, np.asarray(arr))
        except Exception:
            pass
    # ---- 2. GDCM (robust for most compressed MR syntaxes)
    if HAVE_GDCM:
        try:
            import gdcm as _gdcm
            reader = _gdcm.Reader(_gdcm.FileGenerator(str(path)))
            reader.SetDesiredScalarTypeToFloat()
            if reader.Read():
                img = reader.GetImage()
                dims = [img.GetDimension(i) for i in range(3)]
                buf = np.frombuffer(img.GetBuffer(), dtype=np.uint8)
                if buf.size == int(np.prod(dims)):
                    arr = buf.reshape(dims[::-1]).astype(np.float32)
                    return arr
        except Exception:
            pass
    # ---- 3. imageio (handles some JPEG/JPEG2000 payloads)
    if HAVE_IMAGEIO:
        try:
            arr = iio.imread(str(path))
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            pass
    # ---- 4. raw reinterpretation of the encapsulated stream
    if HAVE_PYDICOM:
        try:
            ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=False)
            data = bytes(ds.get("PixelData", b""))
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
            bits = int(getattr(ds, "BitsAllocated", 16) or 16)
            dtype = np.uint16 if bits <= 16 else np.uint32
            if rows and cols:
                arr = np.frombuffer(data, dtype=dtype)
                need = rows * cols
                if arr.size >= need:
                    return arr[:need].reshape(rows, cols).astype(np.float32)
        except Exception:
            pass
    return None


def load_slice_array(path: Path) -> Optional[np.ndarray]:
    """Return one 2D float32 slice, or None if the file is corrupt/unreadable."""
    try:
        if HAVE_PYDICOM:
            ds = pydicom.dcmread(str(path), force=True)
            try:
                arr = np.asarray(ds.pixel_array)
                return _apply_rescale(ds, arr)
            except Exception:
                return _decode_pixels_fallbacks(path)
        return _decode_pixels_fallbacks(path)
    except Exception:
        return None


# ------------------------------------------------------------- series scanning
@dataclass
class SliceMeta:
    path: Path
    index: int = 0                 # position after sorting
    location: float = 0.0          # SliceLocation / IPP z
    instance: int = 0
    plane: int = PLANE_UNKNOWN
    array: Optional[np.ndarray] = None


@dataclass
class SeriesMeta:
    study_uid: str
    series_uid: str
    plane: int = PLANE_UNKNOWN
    num_slices: int = 0
    fluid_sensitive: int = 0
    fat_suppressed: int = 0
    weighted: str = "OTHER"
    description: str = ""
    paths: List[Path] = field(default_factory=list)


class DicomSeriesReader:
    """Fast, thread-safe reader turning a series folder into a sampled volume."""

    def __init__(self, cfg: Config = CFG, max_workers: int = 4) -> None:
        self.cfg = cfg
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._cache_dir = Path(cfg.VOLUME_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- header scan
    @staticmethod
    def scan_series_header(series_dir: Path, study_uid: str) -> Optional[SeriesMeta]:
        """Metadata-only pass over a series (no pixel decode) - cheap & parallel-safe."""
        files = sorted([p for p in series_dir.rglob("*") if p.is_file()])
        if not files:
            return None
        planes: Dict[int, int] = {}
        desc = ""
        fs = fas = 0
        weighted = "OTHER"
        n_hdr = 0
        for f in files[:6]:                     # sample headers only
            if not HAVE_PYDICOM:
                break
            try:
                ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
            except Exception:
                continue
            n_hdr += 1
            desc = desc or str(getattr(ds, "SeriesDescription", "") or "")
            plane = classify_orientation(getattr(ds, "ImageOrientationPatient", None))
            planes[plane] = planes.get(plane, 0) + 1
            meta = parse_sequence_metadata(desc, str(getattr(ds, "SeriesInstanceUID", "")),
                                           str(getattr(ds, "SequenceVariant", "")) +
                                           " " + str(getattr(ds, "ScanningSequence", "")))
            fs = max(fs, meta["fluid_sensitive"])
            fas = max(fas, meta["fat_suppressed"])
            weighted = meta["weighted"] if meta["weighted"] != "OTHER" else weighted
        if n_hdr == 0:
            return None
        plane = max(planes, key=lambda k: planes[k]) if planes else PLANE_UNKNOWN
        return SeriesMeta(
            study_uid=study_uid,
            series_uid=series_dir.name,
            plane=plane,
            num_slices=len(files),
            fluid_sensitive=fs,
            fat_suppressed=fas,
            weighted=weighted,
            description=desc,
            paths=files,
        )

    # ------------------------------------------------------ slice ordering
    @staticmethod
    def order_slices(metas: List[SliceMeta]) -> List[SliceMeta]:
        """Sort slices by physical location; fall back to InstanceNumber then name."""
        def loc_key(m: SliceMeta) -> Tuple[float, int, str]:
            return (m.location if math.isfinite(m.location) else float(m.instance),
                    m.instance, str(m.path))
        if any(abs(m.location) > 1e-6 for m in metas):
            ordered = sorted(metas, key=loc_key)
        else:
            ordered = sorted(metas, key=lambda m: (m.instance, str(m.path)))
        for i, m in enumerate(ordered):
            m.index = i
        return ordered

    # ----------------------------------------------------- single series read
    def _read_one_series(self, series_dir: Path, study_uid: str) -> Optional[Dict[str, Any]]:
        smeta = self.scan_series_header(series_dir, study_uid)
        if smeta is None:
            return None
        metas: List[SliceMeta] = []
        for f in smeta.paths:
            if not HAVE_PYDICOM:
                arr = load_slice_array(f)
                if arr is None:
                    continue
                metas.append(SliceMeta(path=f, array=arr))
                continue
            try:
                ds = pydicom.dcmread(str(f), force=True)
            except Exception:
                continue
            loc = getattr(ds, "SliceLocation", None)
            if loc is None:
                ipp = getattr(ds, "ImagePositionPatient", None)
                loc = float(ipp[2]) if ipp is not None and len(ipp) == 3 else None
            inst = getattr(ds, "InstanceNumber", None)
            metas.append(SliceMeta(
                path=f,
                location=float(loc) if loc is not None else float("nan"),
                instance=int(inst) if inst is not None else 0,
                plane=classify_orientation(getattr(ds, "ImageOrientationPatient", None)),
            ))
        if not metas:
            return None
        metas = self.order_slices(metas)

        # Decode pixels with bounded intra-series parallelism.
        arrays: List[Optional[np.ndarray]] = [None] * len(metas)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            idxs = list(range(len(metas)))
            for i, arr in zip(idxs, pool.map(lambda i: load_slice_array(metas[i].path), idxs)):
                arrays[i] = arr
        valid = [(m, a) for m, a in zip(metas, arrays) if a is not None and a.ndim == 2]
        if len(valid) < 2:                       # degenerate/corrupt series
            return None

        vol = np.stack([a for _, a in valid], axis=0)           # (D, H, W)
        vol = self.resample_depth(vol, self.cfg.NUM_SLICES)     # (K, H, W)
        vol = self.normalize_volume(vol)                         # [0,1] percentile stretch
        vol = self.resize_spatial(vol, self.cfg.IMAGE_SIZE)      # (K, h, w)
        return {
            "volume": vol.astype(np.float32),
            "plane": int(smeta.plane),
            "num_slices": int(len(valid)),
            "series_uid": smeta.series_uid,
            "metadata": {
                "Fluid_Sensitive": int(smeta.fluid_sensitive),
                "Fat_Suppression": int(smeta.fat_suppressed),
                "Plane": int(smeta.plane),
                "Weighted": smeta.weighted,
                "Description": smeta.description,
            },
        }

    # --------------------------------------------------------- normalisation
    def normalize_volume(self, vol: np.ndarray) -> np.ndarray:
        """Percentile-clipped min-max contrast stretch (removes scanner anomalies)."""
        lo = float(np.percentile(vol, self.cfg.PERC_LOW))
        hi = float(np.percentile(vol, self.cfg.PERC_HIGH))
        if hi - lo < 1e-6:
            return np.zeros_like(vol, dtype=np.float32)
        out = (vol - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------- depth resampling
    @staticmethod
    def resample_depth(vol: np.ndarray, k: int) -> np.ndarray:
        """Uniform 1D linspace sampling of D real slices -> exactly K frames."""
        d = vol.shape[0]
        if d == k:
            return vol
        if d == 1:
            return np.repeat(vol, k, axis=0)
        dst = np.linspace(0.0, d - 1.0, k)   # uniform depth sampling positions
        lo = np.floor(dst).astype(np.int64)
        hi = np.minimum(lo + 1, d - 1)
        w = (dst - lo)[..., None, None].astype(np.float32)
        return (vol[lo] * (1.0 - w) + vol[hi] * w).astype(np.float32)

    # ------------------------------------------------------ spatial resizing
    @staticmethod
    def resize_spatial(vol: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        if vol.shape[1:] == (h, w):
            return vol
        if HAVE_CV2:
            out = np.empty((vol.shape[0], h, w), dtype=np.float32)
            for i in range(vol.shape[0]):
                out[i] = cv2.resize(vol[i], (w, h), interpolation=cv2.INTER_AREA)
            return out
        # pure-numpy bilinear fallback
        zi = np.linspace(0, vol.shape[1] - 1, h)
        zj = np.linspace(0, vol.shape[2] - 1, w)
        y0 = np.floor(zi).astype(int); x0 = np.floor(zj).astype(int)
        y1 = np.minimum(y0 + 1, vol.shape[1] - 1); x1 = np.minimum(x0 + 1, vol.shape[2] - 1)
        wy = (zi - y0)[:, None]; wx = (zj - x0)[None, :]
        a = vol[:, y0][:, :, x0]; b = vol[:, y0][:, :, x1]
        c = vol[:, y1][:, :, x0]; d = vol[:, y1][:, :, x1]
        top = a * (1 - wx) + b * wx
        bot = c * (1 - wx) + d * wx
        return (top * (1 - wy) + bot * wy).astype(np.float32)

    # ------------------------------------------------------------ study cache
    def _cache_path(self, study_uid: str) -> Path:
        h = hashlib.md5(study_uid.encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}.npz"

    def load_study(self, study_dirs: Sequence[Path], study_uid: str,
                   planes_wanted: Sequence[int] = (0, 1, 2)) -> Dict[str, Any]:
        """Load up to one dominant series per requested plane for a study.

        Returns dict: volumes (P,K,H,W) float32, plane_ids (P,), meta_flags (P,3),
        mask (P,) 1=real 0=padded-zero. Cached on disk for O(1) epoch re-reads.
        """
        cp = self._cache_path(study_uid)
        if self.cfg.ENABLE_CACHING and cp.exists():
            try:
                z = np.load(cp, allow_pickle=True)
                return {k: z[k] for k in z.files} | {"_cached": True}
            except Exception:
                cp.unlink(missing_ok=True)

        # Collect candidate series across every mounted dir for this study.
        candidates: List[SeriesMeta] = []
        for root in study_dirs:
            if not root.is_dir():
                continue
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and (sub.name.startswith(study_uid) or study_uid in sub.name):
                    m = self.scan_series_header(sub, study_uid)
                    if m is not None:
                        candidates.append(m)

        # Prefer one series per wanted plane; fill remaining slots by slice count.
        chosen: List[SeriesMeta] = []
        used: set = set()
        for pl in planes_wanted:
            best = None
            for c in candidates:
                if c.plane == pl and id(c) not in used:
                    if best is None or c.num_slices > best.num_slices:
                        best = c
            if best is not None:
                chosen.append(best); used.add(id(best))
        for c in sorted(candidates, key=lambda x: -x.num_slices):
            if len(chosen) >= self.cfg.NUM_PLANES:
                break
            if id(c) not in used:
                chosen.append(c); used.add(id(c))

        vols, plane_ids, flags, mask = [], [], [], []
        for slot_pl in planes_wanted:
            got = None
            for c in chosen:
                if c.plane == slot_pl:
                    got = c
                    break
            if got is None:                      # missing anatomical plane -> zero pad
                vols.append(np.zeros((self.cfg.NUM_SLICES, *self.cfg.IMAGE_SIZE), np.float32))
                plane_ids.append(slot_pl)
                flags.append([0, 0, slot_pl])
                mask.append(0.0)
                continue
            rec = self._read_one_series(Path(got.paths[0]).parent, study_uid)
            if rec is None:
                vols.append(np.zeros((self.cfg.NUM_SLICES, *self.cfg.IMAGE_SIZE), np.float32))
                plane_ids.append(slot_pl)
                flags.append([0, 0, slot_pl])
                mask.append(0.0)
            else:
                v = rec["volume"]
                if v.shape[-2:] != tuple(self.cfg.IMAGE_SIZE):
                    v = self.resize_spatial(v, self.cfg.IMAGE_SIZE)
                vols.append(v)
                plane_ids.append(rec["plane"])
                md = rec["metadata"]
                flags.append([md["Fluid_Sensitive"], md["Fat_Suppression"], rec["plane"]])
                mask.append(1.0)

        out = {
            "volumes": np.stack(vols, 0).astype(np.float32),          # (3,K,H,W)
            "plane_ids": np.asarray(plane_ids, dtype=np.int64),       # (3,)
            "meta_flags": np.asarray(flags, dtype=np.float32),        # (3,3)
            "mask": np.asarray(mask, dtype=np.float32),               # (3,)
        }
        if self.cfg.ENABLE_CACHING:
            try:
                np.savez_compressed(cp, **out)
            except Exception:
                pass
        return out


READER = DicomSeriesReader(CFG, max_workers=max(2, CFG.NUM_WORKERS))
print("[dicom] reader ready | pydicom=%s | cache=%s" % (HAVE_PYDICOM, READER._cache_dir))
