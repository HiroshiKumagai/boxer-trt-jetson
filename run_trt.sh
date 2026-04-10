#!/bin/bash
# Run Boxer inference with TensorRT engines.
#
# Usage:
#   bash run_trt.sh --input sample_data/hohen_gen1 --track

set -e

docker run --rm \
  --gpus all \
  --runtime=nvidia \
  -v "$PWD"/boxer:/workspace/boxer \
  -v "$PWD"/sample_data:/workspace/boxer/sample_data \
  -v "$PWD"/ckpts:/workspace/boxer/ckpts \
  -v "$PWD"/output:/workspace/boxer/output \
  -v "$PWD"/trt_engines:/workspace/trt_engines \
  -v "$PWD"/python:/workspace/python \
  -w /workspace/boxer \
  boxer-infer-x86_64 \
  python /workspace/python/run_boxer_trt.py "$@"
