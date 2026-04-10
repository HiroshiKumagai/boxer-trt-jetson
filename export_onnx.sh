#!/bin/bash
# Export Boxer models (OWLv2, DinoV3, BoxerNetCore) to ONNX format.
#
# Usage:
#   bash export_onnx.sh           # export all models
#   bash export_onnx.sh --owl     # OWLv2 only
#   bash export_onnx.sh --dino    # DinoV3 only
#   bash export_onnx.sh --boxernet # BoxerNetCore only

set -e

ARGS="${@:---all}"

docker run --rm \
  --gpus all \
  --runtime=nvidia \
  -v "$PWD"/boxer:/workspace/boxer \
  -v "$PWD"/ckpts:/workspace/boxer/ckpts \
  -v "$PWD"/onnx_weights:/workspace/onnx_weights \
  -v "$PWD"/python:/workspace/python \
  -w /workspace/boxer \
  boxer-convert-x86_64 \
  python /workspace/python/onnx_export.py ${ARGS}
