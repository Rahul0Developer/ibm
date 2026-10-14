# ============================================================
# SECTION 1b - KAGGLE DATA-LAYOUT AUTO-DETECTION & DEPENDENCIES
# ============================================================
# Kaggle mounts competition data under /kaggle/input/<comp>/ and the test set is
# DICOM-only (no reports). We resolve paths defensively so the SAME notebook runs
# in: (a) local dev, (b) attached-sample-submission/debug mode, (c) offline submit.
from __future__ import annotations

import subprocess
from typing import Dict, List


def _pip_install(packages: List[str], quiet: bool = True) -> int:
    """Best-effort pip install (internet available in *train* mode only).

    Wrapped so that a failure (offline submission run) is non-fatal: every consumer
    of these packages has an offline-safe fallback path.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] if quiet else \
          [sys.executable, "-m", "pip", "install"]
    cmd += ["--no-warn-script-location"] + packages
    try:
        return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[deps] pip failed ({exc}); relying on pre-installed/offline wheels")
        return 1


# pydicom            : DICOM parsing
# pylibjpeg-openjpeg : JPEG2000 / JPEG-LS transfer syntaxes (RSNA MRI uses these!)
# pylibjpeg-libjpeg  : JPEG-lossless unpacking
# albumentations     : slice-consistent 2D augmentation
# timm               : ConvNeXt / EfficientNet backbones
_DEPS_OK: Dict[str, bool] = {}
for _pkg in ("pydicom", "pylibjpeg", "pylibjpeg_openjpeg", "pylibjpeg_libjpeg",
             "albumentations", "timm", "sklearn"):
    try:
        __import__(_pkg)
        _DEPS_OK[_pkg] = True
    except Exception:
        _DEPS_OK[_pkg] = False

_missing = [p for p, ok in _DEPS_OK.items() if not ok]
if _missing:
    # Kaggle keeps internet ON for the notebook session; offline submissions simply
    # skip this because all wheels ship in the base image / attached dataset.
    print(f"[deps] attempting install of: {_missing}")
    _pip_install([m if m != "sklearn" else "scikit-learn" for m in _missing])
else:
    print("[deps] all required packages already importable")

# Re-import critical third-party modules now that deps are resolved.
try:
    import pydicom
    from pydicom import dcmread
    from pydicom.uid import UID
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False
    print("[deps] WARNING: pydicom unavailable -> DICOM decoding will use gdcm/imageio fallback")

try:
    import imageio.v3 as iio  # type: ignore
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import gdcm  # noqa: F401
    HAVE_GDCM = True
except Exception:
    HAVE_GDCM = False

print(f"[deps] pydicom={HAVE_PYDICOM} gdcm={HAVE_GDCM} imageio={HAVE_IMAGEIO} cv2={HAVE_CV2}")


# ------------------------------------------------------------------ path logic
class DataLayout:
    """Resolve competition file/dir locations across Kaggle mount conventions."""

    CANDIDATE_ROOTS: List[str] = [
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/rsna-2025-knee-abnormality-detection",
        "/kaggle/input/rsna*",
        CFG.DATA_ROOT,
        "./input",
        ".",
    ]

    def __init__(self, cfg: Config = CFG) -> None:
        self.cfg = cfg
        self.root: Path = self._find_root()
        self.train_csv: Optional[Path] = self._first_existing(
            ["train.csv", "train_sample.csv"], self.root)
        self.test_csv: Optional[Path] = self._first_existing(
            ["test.csv", "sample_submission.csv"], self.root)
        self.submission_csv: Optional[Path] = self._first_existing(
            ["sample_submission.csv"], self.root)
        self.train_dirs: List[Path] = self._series_dirs(["train_series", "train_dicoms", "train"])
        self.test_dirs: List[Path] = self._series_dirs(["test_series", "test_dicoms", "test"])
        self.reports_parquet: Optional[Path] = self._first_existing(
            ["train_reports.parquet", "reports.parquet"], self.root)

    @staticmethod
    def _glob_roots() -> List[Path]:
        import glob as _glob
        out: List[Path] = []
        for pat in DataLayout.CANDIDATE_ROOTS:
            out.extend(sorted(Path(p) for p in _glob.glob(pat) if Path(p).is_dir()))
        return out

    def _find_root(self) -> Path:
        for cand in self._glob_roots():
            names = {p.name.lower() for p in cand.iterdir()} if cand.is_dir() else set()
            if any("train" in n or "test" in n or n.endswith(".csv") for n in names):
                print(f"[layout] data root -> {cand}")
                return cand
        fallback = Path(self.cfg.DATA_ROOT)
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[layout] no competition mount found -> using {fallback} (synthetic demo mode)")
        return fallback

    @staticmethod
    def _first_existing(names: Sequence[str], root: Path) -> Optional[Path]:
        for n in names:
            p = root / n
            if p.exists():
                return p
        # one-level deeper search (nested kaggle datasets)
        for sub in sorted(root.glob("*/"))[:12]:
            for n in names:
                p = sub / n
                if p.exists():
                    return p
        return None

    def _series_dirs(self, names: Sequence[str]) -> List[Path]:
        found: List[Path] = []
        for n in names:
            p = self.root / n
            if p.is_dir():
                found.append(p)
        if not found:  # nested layout
            for sub in sorted(self.root.glob("*/"))[:12]:
                for n in names:
                    p = sub / n
                    if p.is_dir():
                        found.append(p)
        return found

    @property
    def has_real_data(self) -> bool:
        return bool(self.train_csv or self.train_dirs or self.test_dirs)

    def describe(self) -> str:
        return json.dumps(
            {
                "root": str(self.root),
                "train_csv": str(self.train_csv) if self.train_csv else None,
                "test_csv": str(self.test_csv) if self.test_csv else None,
                "train_dirs": [str(p) for p in self.train_dirs],
                "test_dirs": [str(p) for p in self.test_dirs],
                "has_real_data": self.has_real_data,
            },
            indent=2,
        )


LAYOUT = DataLayout()
print(LAYOUT.describe())

IS_KAGGLE: bool = Path("/kaggle/input").exists()
# Submission inference mode == test.csv present with >3 rows and NO train labels needed.
SUBMISSION_MODE: bool = (
    LAYOUT.test_csv is not None
    and LAYOUT.train_csv is None
    and len(pd.read_csv(LAYOUT.test_csv)) > 3
)
print(f"[layout] IS_KAGGLE={IS_KAGGLE} SUBMISSION_MODE={SUBMISSION_MODE}")
