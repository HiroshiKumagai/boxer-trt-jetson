#!/bin/bash
# Run Boxer inference with ONNX Runtime.
#
# Usage:
#   bash run_onnx.sh --input input

set -e

docker run --rm \
  --gpus all \
  --runtime=nvidia \
  -v "$PWD"/boxer:/workspace/boxer \
  -v "$PWD"/input:/workspace/boxer/input \
  -v "$PWD"/boxer/ckpts:/workspace/boxer/ckpts \
  -v "$PWD"/output:/workspace/boxer/output \
  -v "$PWD"/onnx_weights:/workspace/onnx_weights \
  -v "$PWD"/python:/workspace/python \
  -w /workspace/boxer \
  boxer-infer-jetson \
  python3 /workspace/python/run_boxer_onnx.py "$@"
