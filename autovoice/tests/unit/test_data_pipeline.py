"""Unit tests for the Phase-2 data pipeline (no network, no GCP)."""

from __future__ import annotations

import pandas as pd
import pytest

from autovoice.data.clean import _to_bool_01, clean, normalize_text
from autovoice.data.download import REQUIRED_COLUMNS, _read_csv


def _fake_raw(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "cmplid": f"{i:06d}",
                "make": "FORD" if i % 2 else "NISSAN",
                "model": "FUSION",
                "model_year": str(2018 + (i % 5)),
                "comp_desc": "ELECTRICAL SYSTEM",
                "complaint_desc": (
                    f"The vehicle stalled while driving narrative number {i} "
                    "with a long enough description to pass the minimum length."
                ),
                "crash": "1" if i % 10 == 0 else "0",
                "fire": "Y" if i % 15 == 0 else "N",
                "injured": "0",
                "deaths": "0",
                "date_added": f"2026-0{(i % 9) + 1}-15",
            }
        )
    return pd.DataFrame(rows)


class TestNormalizeText:
    def test_collapses_whitespace_and_smart_quotes(self):
        raw = "engine\u2019s  failed\u2014again\tand\tstalled"
        assert normalize_text(raw) == "engine's failed-again and stalled"

    def test_strips_boilerplate(self):
        out = normalize_text("Customer states that the brakes failed suddenly today")
        assert "customer states" not in out.lower()
        assert "brakes failed" in out

    def test_non_string_returns_empty(self):
        assert normalize_text(None) == ""  # type: ignore[arg-type]


class TestIndicators:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("1", 1), ("y", 1), ("YES", 1), ("true", 1), ("0", 0), ("n", 0), ("", 0)],
    )
    def test_to_bool_01(self, value, expected):
        got = _to_bool_01(pd.Series([value]))[0]
        assert got == expected


class TestSchemaValidation:
    def test_missing_column_raises(self):
        bad = pd.DataFrame({"cmplid": ["1"]})  # missing most required fields
        with pytest.raises(RuntimeError, match="missing required NHTSA columns"):
            _read_csv(bad.to_csv(index=False).encode())

    def test_read_csv_accepts_valid_schema(self):
        df = _read_csv(_fake_raw(3).to_csv(index=False).encode())
        assert set(df.columns) >= REQUIRED_COLUMNS


class TestClean:
    def test_drops_short_narratives(self):
        df = _fake_raw(10)
        df.loc[0, "complaint_desc"] = "too short"
        out = clean(df)
        assert len(out) == 9
        assert "narrative" in out.columns
        assert "complaint_desc" not in out.columns

    def test_dedupes_on_identity(self):
        df = pd.concat([_fake_raw(10)] * 2, ignore_index=True)
        out = clean(df)
        assert len(out) == 10

    def test_caps_rows(self):
        out = clean(_fake_raw(30), max_rows=5)
        assert len(out) == 5

    def test_indicators_are_ints(self):
        out = clean(_fake_raw(20))
        for col in ("crash", "fire", "injured", "deaths"):
            assert out[col].dtype.kind == "i"
