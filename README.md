# boxer-trt-jetson

Run [Boxer](https://github.com/facebookresearch/boxer) 3D object detection on NVIDIA Jetson (JetPack 6.x / aarch64) with GPU-accelerated inference using ONNX Runtime and TensorRT.

Boxer detects and tracks 3D bounding boxes from a single RGB camera using OWLv2 (2D detection), DinoV3 (feature extraction), and BoxerNetCore (3D estimation). This repository provides tooling to export all three models to ONNX and run them on Jetson with the DinoV3 stage accelerated by TensorRT through ONNX Runtime's `TensorrtExecutionProvider`.

> **Note:** JetPack 5.x (R35) is not supported. JetPack 6.x (R36) only.

---

## Benchmark results

Measured on Jetson Orin NX with the `hohen_gen1` sequence (499 frames, `--track`).

| Backend | Total | Per frame | OWLv2 | BoxerNet | GPU avg | RAM peak |
|---------|-------|-----------|-------|----------|---------|----------|
| ONNX Runtime (all CUDA) | 754 s | 1.51 s/f | 511 ms | 461 ms | 59% | 27.9 GB |
| ONNX Runtime (DinoV3=TRT FP32) | 626 s | 1.25 s/f | 412 ms | 388 ms | 51% | 26.8 GB |
| ONNX Runtime (DinoV3=TRT FP16) | **538 s** | **1.08 s/f** | 412 ms | **234 ms** | 26% | 29.2 GB |

FP16 produces ~7% fewer 3D detections per frame than FP32 due to reduced precision in DinoV3 feature extraction. All three variants produce correct detections; FP32 is recommended for accuracy-sensitive use cases.

---

## Quick start

### Requirements

- NVIDIA Jetson with JetPack 6.x (L4T R36.x, TensorRT 10.3)
- Docker with NVIDIA Container Toolkit
- ~50 GB free disk space (Docker images + model weights)

### 1. Clone this repository and Boxer

```bash
git clone https://github.com/HiroshiKumagai/boxer-trt-jetson
cd boxer-trt-jetson
git clone https://github.com/facebookresearch/boxer boxer
```

### 2. Download Boxer checkpoints

```bash
cd boxer
bash scripts/download_ckpts.sh
cd ..
```

### 3. Download sample data (optional)

```bash
bash boxer/scripts/download_aria_data.sh hohen_gen1
mkdir -p sample_data
mv hohen_gen1 sample_data/
```

### 4. Build Docker images

```bash
# ONNX export + TRT engine builder
docker build -f docker/Dockerfile.convert.jetson -t boxer-convert-jetson .

# Inference runtime (builds projectaria_tools from source, ~25 min)
docker build -f docker/Dockerfile.infer.jetson -t boxer-infer-jetson .
```

### 5. Export models to ONNX

```bash
bash export_onnx.sh            # all models
bash export_onnx.sh --owl      # OWLv2 only
bash export_onnx.sh --dino     # DinoV3 only (also runs fix_dinov3_onnx.py)
bash export_onnx.sh --boxernet # BoxerNetCore only
```

Output in `onnx_weights/`: `owlv2_vision_detector.onnx`, `owlv2_meta.pt`, `dinov3.onnx`, `boxernet_core.onnx`

### 6. Run inference

```bash
# ONNX Runtime (DinoV3 accelerated via TensorRT, OWLv2 + Core on CUDA)
bash run_onnx.sh --input sample_data/hohen_gen1 --track
```

On first run, ORT compiles and caches the DinoV3 TRT engine under `trt_cache/`. Subsequent runs reuse the cache.

### 7. Benchmark with GPU monitoring

```bash
bash benchmark.sh onnx --input sample_data/hohen_gen1 --track
```

Reports total wall-clock time, GPU utilization (from `tegrastats`), and RAM peak.

### Optional — DinoV3 FP16

Edit [python/run_boxer_onnx.py](python/run_boxer_onnx.py) and set `trt_fp16_enable: True`. FP16 is ~14% faster than FP32 but reduces 3D detection count by ~7%.

---

## Design decisions

### Why not custom TRT engines?

Three TRT engines were built from the exported ONNX files, but all three had problems that made them unsuitable for use:

**OWLv2 — FP16 produces NaN:**
OWLv2 is CLIP-based and contains a learned `logit_scale` parameter applied as `exp(logit_scale)`. This value exceeds the FP16 range (±65504) and produces NaN in all logits, resulting in zero detections.

**OWLv2 — TRT cannot compile at all:**
`TensorrtExecutionProvider` rejects the OWLv2 ONNX with:
```
Could not find any implementation for node /embeddings/patch_embedding/Conv.
```
The depthwise Conv configuration used in the OWLv2 patch embedding has no TRT kernel for this combination of parameters on Jetson.

**BoxerNetCore — Dynamic M dimension causes per-frame recompilation:**
The input `bb2d_norm` has shape `(1, M, 4)` where M is the number of 2D detections and changes every frame. TRT builds a separate engine for each unique M value. Each build takes ~27 seconds, making the first occurrence of any new M value stall the pipeline. Unusable in practice.

**DinoV3 custom FP32 engine — Feature values diverge:**
The engine builds and runs without error, but the output features have large numerical differences from ONNX Runtime (max abs diff 4.88 on a random input vs 0.02 for the ORT TRT provider). This causes 3D detections to drop to near zero. The likely cause is that `fix_dinov3_onnx.py` inlines the rope_embed `If` nodes by hand — the resulting graph structure is optimized differently by TRT's kernel selector compared to the original graph.

**Solution — Use ORT `TensorrtExecutionProvider` for DinoV3 only:**
ONNX Runtime's `TensorrtExecutionProvider` compiles DinoV3 correctly: it automatically partitions the graph into TRT-compatible subgraphs and falls back to CUDA for unsupported nodes. Output values match the CUDA-only baseline. OWLv2 and BoxerNetCore remain on `CUDAExecutionProvider` to avoid the issues above.

The custom TRT build scripts (`build_engine.sh`, `run_trt.sh`) are retained for reference.

### Why a multi-stage Docker build for inference?

`projectaria_tools` is required for reading Aria VRS files (`hohen_gen1` and other Aria sequences). It has no pre-built aarch64 wheel on PyPI and must be compiled from source, including the Ocean C++ library (~25 minutes, ~3 GB of build artifacts).

A multi-stage build keeps the build toolchain out of the final runtime image:

- **Stage 1 (builder):** installs cmake, ninja, boost, ffmpeg dev headers, and all other build dependencies; compiles `projectaria_tools` from source with `-flax-vector-conversions` to work around a gcc-11 NEON signed/unsigned vector type error in `FrameConverter.h`; produces a Python wheel.
- **Stage 2 (runtime):** copies only the compiled wheel, installs runtime shared libraries (`libboost-filesystem`, `libfmt8`, `ffmpeg`, etc.), and discards all build tooling.

### Why `onnxruntime-gpu` from `pypi.jetson-ai-lab.io`?

The standard `onnxruntime` package on PyPI is CPU-only and reports `CUDAExecutionProvider` as unavailable on Jetson. The Jetson-specific GPU build with CUDA and TensorRT support is published at:

```
https://pypi.jetson-ai-lab.io/jp6/cu128/
```

This wheel enables both `CUDAExecutionProvider` and `TensorrtExecutionProvider`.

### Why `numpy<2`?

PyTorch 2.7 in `dustynv/pytorch:2.7-r36.4.0` was compiled against NumPy 1.x. NumPy 2.x is binary-incompatible and causes `RuntimeError: Numpy is not available` when importing PyTorch. The base image ships NumPy 2.x, and some dependencies (`rerun-sdk`, used by `projectaria_tools`) re-upgrade to 2.x. Installing `numpy<2` last with `--force-reinstall` pins it correctly.

### Why DinoV3 ONNX has no height/width dynamic axes?

The initial export used `{0:"batch", 2:"height", 3:"width"}` dynamic axes for DinoV3. This caused the ONNX exporter to produce symbolic-shape-dependent `If` nodes for the rope position embedding, where each branch has a different output shape. TRT cannot parse `If` nodes with mismatched branch output shapes. Removing height and width dynamic axes eliminates this problem. The `fix_dinov3_onnx.py` script additionally inlines any remaining `If` nodes as a safety measure.

---

## Repository structure

```
.
├── docker/
│   ├── Dockerfile.convert.jetson   # ONNX export + TRT engine builder
│   └── Dockerfile.infer.jetson     # Multi-stage inference runtime
├── python/
│   ├── onnx_export.py              # Export OWLv2, DinoV3, BoxerNetCore to ONNX
│   ├── fix_dinov3_onnx.py          # Inline rope_embed If nodes for TRT compatibility
│   ├── build_engines.py            # Build custom TRT engines (reference only)
│   ├── run_boxer_onnx.py           # ONNX Runtime inference (monkey-patches run_boxer.py)
│   └── run_boxer_trt.py            # Custom TRT inference (reference only)
├── export_onnx.sh                  # Export + auto-fix DinoV3
├── build_engine.sh                 # Build custom TRT engines (reference only)
├── run_onnx.sh                     # Run ONNX Runtime inference
├── run_trt.sh                      # Run custom TRT inference (reference only)
└── benchmark.sh                    # Run inference with tegrastats GPU monitoring
```

---

## License

MIT. The Boxer model is licensed under CC-BY-NC; see the [Boxer repository](https://github.com/facebookresearch/boxer) for details.
