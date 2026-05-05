#!/bin/bash
# Export Boxer models (OWLv2, DinoV3, BoxerNetCore) to ONNX format.
#
# Usage:
#   bash export_onnx.sh           # export all models
#   bash export_onnx.sh --owl     # OWLv2 only
#   bash export_onnx.sh --dino    # DinoV3 only
#   bash export_onnx.sh --boxernet # BoxerNetCore only
#
# Note: DinoV3 ONNX contains rope_embed/If nodes that TensorRT cannot parse.
#       fix_dinov3_onnx.py is invoked automatically after export to inline them.

set -e

ARGS="${@:---all}"

# Step 1: ONNX export (CPU only — no --gpus flag needed)
docker run --rm \
  -v "$PWD"/boxer:/workspace/boxer \
  -v "$PWD"/boxer/ckpts:/workspace/boxer/ckpts \
  -v "$PWD"/onnx_weights:/workspace/onnx_weights \
  -v "$PWD"/python:/workspace/python \
  -w /workspace/boxer \
  boxer-convert-jetson \
  python3 /workspace/python/onnx_export.py --simplify ${ARGS}

# Step 2: inline DinoV3 rope_embed/If nodes (only when --dino or --all)
if echo "${ARGS}" | grep -qE '(--all|--dino)'; then
  echo "==> Fixing rope_embed/If nodes in dinov3.onnx..."
  docker run --rm \
    -v "$PWD"/onnx_weights:/workspace/onnx_weights \
    -v "$PWD"/python:/workspace/python \
    boxer-convert-jetson \
    python3 /workspace/python/fix_dinov3_onnx.py
fi
