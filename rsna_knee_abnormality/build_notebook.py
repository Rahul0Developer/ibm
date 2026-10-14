#!/usr/bin/env python3
"""Notebook assembler for the RSNA Knee Abnormality Detection Kaggle solution.

Reads every ``cells/`` fragment in numeric order and serialises them into a single
valid ``.ipynb`` file (markdown + code cells) using ``nbformat``.

Usage::

    python build_notebook.py            # writes rsna_knee_abnormality_2p5d_asl.ipynb
"""
from __future__ import annotations

import glob
import json
import os
from typing import List, Tuple

import nbformat as nbf

CELLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells")
OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rsna_knee_abnormality_2p5d_asl.ipynb",
)


def _split_kind(fname: str) -> Tuple[str, str]:
    """Return ``(kind, body)`` for a cell fragment file name.

    Naming convention: ``NNN_type.md`` / ``NNN_type.py`` where type is
    ``md`` (markdown) or ``code`` (python).
    """
    base = os.path.basename(fname)
    parts = base.split("_")
    kind = "markdown" if parts[1].startswith("md") else "code"
    with open(fname, "r", encoding="utf-8") as handle:
        body = handle.read().rstrip("\n")
    return kind, body


def main() -> None:
    files: List[str] = sorted(
        glob.glob(os.path.join(CELLS_DIR, "*.md")) + glob.glob(os.path.join(CELLS_DIR, "*.py"))
    )
    if not files:
        raise SystemExit(f"No cell fragments found in {CELLS_DIR}")

    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "jupytext": {"formats": "ipynb,py:percent"},
        }
    )

    cells = []
    for path in files:
        kind, body = _split_kind(path)
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(body))
        else:
            cell = nbf.v4.new_code_cell(body)
            cell.execution_count = None
            cells.append(cell)

    nb.cells = cells
    nbf.validate(nb)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        nbf.write(nb, handle)
    print(f"Wrote {OUT_PATH} with {len(nb.cells)} cells ({json.dumps({'kb': round(os.path.getsize(OUT_PATH)/1024, 1)})})")


if __name__ == "__main__":
    main()
