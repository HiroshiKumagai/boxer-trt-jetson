#!/bin/bash
# Build TensorRT engines from ONNX models.
#
# Usage:
#   bash build_engine.sh          # build all models in FP16 (default)
#   bash build_engine.sh --fp32   # build all models in FP32
#   bash build_engine.sh --owl    # OWLv2 only
#   bash build_engine.sh --dino   # DinoV3 only
#   bash build_engine.sh --boxernet # BoxerNetCore only
#
# Note: TRT engines are platform-specific (aarch64).
#       x86_64 で書き出した .onnx をそのまま使えるが、
#       .plan エンジンは必ず Jetson 上で再ビルドすること。

set -e

ARGS="${@:---all}"

docker run --rm \
  --gpus all \
  --runtime=nvidia \
  -v "$PWD"/onnx_weights:/workspace/onnx_weights \
  -v "$PWD"/trt_engines:/workspace/trt_engines \
  -v "$PWD"/python:/workspace/python \
  -w /workspace \
  boxer-convert-jetson \
  python3 /workspace/python/build_engines.py ${ARGS}
