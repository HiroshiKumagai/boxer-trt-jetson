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
# Note: TRT engines are platform-specific.
#       Rebuild on each target platform (x86_64 / aarch64 Jetson).

set -e

ARGS="${@:---all}"

docker run --rm \
  --gpus all \
  --runtime=nvidia \
  -v "$PWD"/onnx_weights:/workspace/onnx_weights \
  -v "$PWD"/trt_engines:/workspace/trt_engines \
  -v "$PWD"/python:/workspace/python \
  -w /workspace \
  boxer-convert-x86_64 \
  python /workspace/python/build_engines.py ${ARGS}
