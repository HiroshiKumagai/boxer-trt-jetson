#!/bin/bash
# Export Boxer models (OWLv2, DinoV3, BoxerNetCore) to ONNX format.
#
# Usage:
#   bash export_onnx.sh           # export all models
#   bash export_onnx.sh --owl     # OWLv2 only
#   bash export_onnx.sh --dino    # DinoV3 only
#   bash export_onnx.sh --boxernet # BoxerNetCore only
#
# Note: DinoV3 の ONNX には TensorRT が解釈できない rope_embed/If ノードが含まれる。
#       エクスポート後に fix_dinov3_onnx.py で自動修正します。

set -e

ARGS="${@:---all}"

# Step 1: ONNX エクスポート
docker run --rm \
  --gpus all \
  --runtime=nvidia \
  -v "$PWD"/boxer:/workspace/boxer \
  -v "$PWD"/boxer/ckpts:/workspace/boxer/ckpts \
  -v "$PWD"/onnx_weights:/workspace/onnx_weights \
  -v "$PWD"/python:/workspace/python \
  -w /workspace/boxer \
  boxer-convert-jetson \
  python3 /workspace/python/onnx_export.py --simplify ${ARGS}

# Step 2: DinoV3 の rope_embed/If ノードを修正 (--dino or --all の場合のみ)
if echo "${ARGS}" | grep -qE '(--all|--dino)'; then
  echo "==> Fixing rope_embed/If nodes in dinov3.onnx..."
  docker run --rm \
    --runtime=nvidia \
    -v "$PWD"/onnx_weights:/workspace/onnx_weights \
    -v "$PWD"/python:/workspace/python \
    boxer-convert-jetson \
    python3 /workspace/python/fix_dinov3_onnx.py
fi
