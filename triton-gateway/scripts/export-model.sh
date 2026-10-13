#!/usr/bin/env bash
# Export the Hugging Face transformer to ONNX and stage it in the Triton model
# repository. On machines WITHOUT an NVIDIA GPU (e.g. most laptops) this runs a
# CPU export automatically — see docs/LOCAL-DEVELOPMENT.md for the tradeoffs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/model_pipeline/model_repository/text_embeddings/1/model.onnx"

# Auto-detect CUDA so the script works out-of-the-box on GPU-less laptops.
DEVICE="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then DEVICE="cuda"; fi

if [ ! -d "${ROOT}/model_pipeline/.venv" ]; then
  echo ">> Creating virtualenv and installing pipeline deps"
  python3 -m venv "${ROOT}/model_pipeline/.venv"
  "${ROOT}/model_pipeline/.venv/bin/pip" install -q -r "${ROOT}/model_pipeline/requirements.txt"
fi
PY="${ROOT}/model_pipeline/.venv/bin/python"

echo ">> Exporting ONNX (device=${DEVICE}, dynamic batch + sequence axes)"
"${PY}" "${ROOT}/model_pipeline/export_onnx.py" --output "${OUT}" --device "${DEVICE}"

echo ">> Done. Model staged at ${OUT}"
ls -lh "$(dirname "${OUT}")"
