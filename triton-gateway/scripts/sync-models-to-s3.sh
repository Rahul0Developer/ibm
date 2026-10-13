#!/usr/bin/env bash
# Upload the exported Triton model repository to S3 for the init-container pattern.
# Requires: aws CLI v2 configured (or IRSA when run inside the cluster).
# Usage: ./scripts/sync-models-to-s3.sh s3://your-model-bucket/text_embeddings/
set -euo pipefail

DEST="${1:?usage: $0 s3://bucket/prefix/}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/model_pipeline/model_repository/text_embeddings/"

if [ ! -f "${SRC}1/model.onnx" ]; then
  echo "ERROR: ${SRC}1/model.onnx not found — run scripts/export-model.sh first." >&2
  exit 1
fi

command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI not installed." >&2; exit 1; }

echo ">> Syncing ${SRC} -> ${DEST}"
aws s3 sync "${SRC}" "${DEST}" --exclude ".gitkeep"
echo ">> Done. Next rollout of triton-serving will pick up the new version."
