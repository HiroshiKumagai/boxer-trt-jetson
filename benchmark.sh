#!/bin/bash
# Benchmark Boxer ONNX Runtime inference (Jetson) with tegrastats GPU monitoring.
#
# Usage:
#   bash benchmark.sh --input input
#
# Reports:
#   - Wall-clock inference time
#   - GPU utilization (% from tegrastats)
#   - System RAM peak (MB from tegrastats)

set -e

INFER_ARGS="$@"

TSTAT_LOG="/tmp/tegrastats_boxer_onnx.log"
REPORT_LOG="/tmp/boxer_onnx_report.txt"

echo "=== Boxer ONNX Benchmark ==="
echo "Args: ${INFER_ARGS}"
echo ""

# Start tegrastats (100 ms interval)
tegrastats --interval 100 > "${TSTAT_LOG}" 2>&1 &
TSTAT_PID=$!
echo "tegrastats PID: ${TSTAT_PID}  (log: ${TSTAT_LOG})"

# Run inference and time it
START_TS=$(date +%s%N)

docker run --rm \
  --gpus all \
  --runtime=nvidia \
  -v "$PWD"/boxer:/workspace/boxer \
  -v "$PWD"/input:/workspace/boxer/input \
  -v "$PWD"/boxer/ckpts:/workspace/boxer/ckpts \
  -v "$PWD"/output:/workspace/boxer/output \
  -v "$PWD"/onnx_weights:/workspace/onnx_weights \
  -v "$PWD"/trt_cache:/workspace/trt_cache \
  -v "$PWD"/python:/workspace/python \
  -e ORT_TENSORRT_ENGINE_CACHE_ENABLE=1 \
  -e ORT_TENSORRT_CACHE_PATH=/workspace/trt_cache \
  -e BOXER_DISABLE_OPTIMIZED_CACHE \
  -e BOXER_DISABLE_PERSISTENT_BUF \
  -e BOXER_VIZ_ASYNC \
  -e BOXER_CUDA_MEM_LIMIT_GB \
  -e BOXER_TRT_WORKSPACE_MB \
  -e BOXER_CORE_ARENA_STRICT \
  -e BOXER_MEM_PROFILE \
  -e BOXER_DINO_FP16 \
  -w /workspace/boxer \
  boxer-infer-jetson \
  python3 /workspace/python/run_boxer_onnx.py ${INFER_ARGS}

END_TS=$(date +%s%N)
ELAPSED_MS=$(( (END_TS - START_TS) / 1000000 ))

# Stop tegrastats
kill ${TSTAT_PID} 2>/dev/null || true
sleep 0.3

# ---- Parse tegrastats log ----
# tegrastats line example (JetPack 6 / L4T R36):
#   04-11-2026 12:00:00 RAM 3000/7672MB ... GR3D_FREQ 45%@[612] ...
# Portable across GNU grep / ugrep: use awk for tag-based field extraction.
GPU_UTIL_MAX=$(awk '{for(i=1;i<=NF;i++) if($i=="GR3D_FREQ"){gsub("%","",$(i+1)); split($(i+1),a,"@"); print a[1]}}' "${TSTAT_LOG}" 2>/dev/null | sort -n | tail -1 || echo "N/A")
GPU_UTIL_AVG=$(awk '{for(i=1;i<=NF;i++) if($i=="GR3D_FREQ"){gsub("%","",$(i+1)); split($(i+1),a,"@"); print a[1]}}' "${TSTAT_LOG}" 2>/dev/null | awk '{s+=$1;n++} END{if(n>0) printf "%.0f",s/n; else print "N/A"}' || echo "N/A")
RAM_MAX=$(awk '{for(i=1;i<=NF;i++) if($i=="RAM"){split($(i+1),a,"/"); print a[1]}}' "${TSTAT_LOG}" 2>/dev/null | sort -n | tail -1 || echo "N/A")
RAM_TOTAL=$(awk '{for(i=1;i<=NF;i++) if($i=="RAM"){split($(i+1),a,"/"); sub("MB","",a[2]); print a[2]; exit}}' "${TSTAT_LOG}" 2>/dev/null || echo "N/A")

# ---- Report ----
{
  echo "============================================"
  echo "  Boxer ONNX Benchmark Results"
  echo "============================================"
  echo "  Total wall-clock time : ${ELAPSED_MS} ms"
  echo "  GPU utilization (max) : ${GPU_UTIL_MAX}%"
  echo "  GPU utilization (avg) : ${GPU_UTIL_AVG}%"
  echo "  System RAM peak       : ${RAM_MAX} / ${RAM_TOTAL} MB"
  echo "============================================"
  echo "  tegrastats log        : ${TSTAT_LOG}"
} | tee "${REPORT_LOG}"
