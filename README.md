# boxer-trt-jetson

Run [Boxer](https://github.com/facebookresearch/boxer) 3D object detection on NVIDIA Jetson (JetPack 6.x / aarch64) with GPU-accelerated inference using ONNX Runtime and TensorRT.

Boxer detects and tracks 3D bounding boxes from a single RGB camera using OWLv2 (2D detection), DinoV3 (feature extraction), and BoxerNetCore (3D estimation). This repository provides tooling to export all three models to ONNX and run them on Jetson with the DinoV3 stage accelerated by TensorRT through ONNX Runtime's `TensorrtExecutionProvider`.

---

## Benchmark results

Measured on Jetson Orin NX with a 499-frame Aria RGB sequence.

| Backend | Total | Per frame | OWLv2 | BoxerNet | GPU avg | RAM peak |
|---------|-------|-----------|-------|----------|---------|----------|
| ONNX Runtime (all CUDA) | 754 s | 1.51 s/f | 511 ms | 461 ms | 59% | 27.9 GB |
| ONNX Runtime (DinoV3=TRT FP32) | 626 s | 1.25 s/f | 412 ms | 388 ms | 51% | 26.8 GB |
| ONNX Runtime (DinoV3=TRT FP16) | **538 s** | **1.08 s/f** | 412 ms | **234 ms** | 26% | 29.2 GB |

FP16 produces ~7% fewer 3D detections per frame than FP32 due to reduced precision in DinoV3 feature extraction. All three variants produce correct detections; FP32 is recommended for accuracy-sensitive use cases.

---

## Quick start

### Requirements

- NVIDIA Jetson with JetPack 6.x (L4T R36.x, TensorRT 10.3). JetPack 5.x (R35) is not supported.
- Docker with NVIDIA Container Toolkit.
- ~50 GB free disk space (Docker images + model weights).

Validated on: **Jetson AGX Orin Developer Kit** (32 GB unified memory), **L4T R36.4.7** (JetPack 6.2.x).

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

### 3. Build Docker images

```bash
# ONNX export image
docker build -f docker/Dockerfile.convert.jetson -t boxer-convert-jetson .

# Inference runtime
docker build -f docker/Dockerfile.infer.jetson -t boxer-infer-jetson .
```

### 4. Export models to ONNX

```bash
bash export_onnx.sh            # all models
bash export_onnx.sh --owl      # OWLv2 only
bash export_onnx.sh --dino     # DinoV3 only (also runs fix_dinov3_onnx.py)
bash export_onnx.sh --boxernet # BoxerNetCore only
```

Output in `onnx_weights/`: `owlv2_vision_detector.onnx`, `owlv2_meta.pt`, `dinov3.onnx`, `boxernet_core.onnx`.

### 5. Run inference

The repository ships with a 10-frame sample dataset under `input/`, so you can run inference immediately:

```bash
bash run_onnx.sh --input input
```

On first run, ORT compiles and caches the DinoV3 TRT engine under `trt_cache/`. Subsequent runs reuse the cache.

Outputs are written under `output/`:

| File | Contents |
|---|---|
| `boxer_3dbbs.csv` | Per-frame 3D detections in world coordinates. |
| `boxer_3dbbs_tracked.csv` | 3D OBBs after tracker association across frames. |
| `owl_2dbbs.csv` | Per-frame 2D detections from OWLv2. |
| `boxer_viz/boxer_viz_NNNNN.jpg` | 3-panel visualization per frame: 2D detections / 3D detections / 3D tracks. |

### 6. (Optional) Benchmark with GPU monitoring

```bash
bash benchmark.sh --input input
```

Wraps `run_onnx.sh` with `tegrastats` and reports total wall-clock time, GPU utilization, and RAM peak.

### Optional — DinoV3 FP16

Edit [python/run_boxer_onnx.py](python/run_boxer_onnx.py) and set `trt_fp16_enable: True`. FP16 is ~14% faster than FP32 but reduces 3D detection count by ~7%.

### Optional — RAM-constrained tuning

The default configuration prioritizes throughput. To stay under ~17 GB peak RAM at the cost of ~5% throughput, set these env vars:

```bash
BOXER_DISABLE_PERSISTENT_BUF=1 \
BOXER_CUDA_MEM_LIMIT_GB=4 \
BOXER_TRT_WORKSPACE_MB=256 \
BOXER_CORE_ARENA_STRICT=1 \
  bash benchmark.sh --input input
```

| env var | Effect |
|---|---|
| `BOXER_DISABLE_PERSISTENT_BUF=1` | Disable OWL persistent output buffers (saves ~10 GB RAM at the cost of one extra sync point per frame). |
| `BOXER_CUDA_MEM_LIMIT_GB=4` | Cap each ORT CUDA EP's allocator at 4 GB. |
| `BOXER_TRT_WORKSPACE_MB=256` | Cap the DinoV3 TRT engine's build-time workspace at 256 MB. No inference-time impact. |
| `BOXER_CORE_ARENA_STRICT=1` | Force `kSameAsRequested` arena extension on the Core CUDA EP. |
| `BOXER_VIZ_ASYNC=1` | Run `imencode` + per-frame JPG write on a background thread. |

---

## Design decisions

### Why ORT `TensorrtExecutionProvider` instead of standalone TRT engines?

DinoV3 is the only stage that benefits from TRT. OWLv2 and BoxerNetCore both have blockers for standalone TRT:

- **OWLv2** — depthwise patch-embedding Conv has no TRT kernel on Jetson (`Could not find any implementation for node /embeddings/patch_embedding/Conv`). FP16 also overflows OWLv2's `exp(logit_scale)` and emits NaN logits.
- **BoxerNetCore** — `bb2d_norm` has dynamic shape `(1, M, 4)` where M changes every frame; TRT recompiles per unique M (~27 s each), stalling the pipeline.

ORT's `TensorrtExecutionProvider` partitions DinoV3 into TRT-compatible subgraphs and falls back to CUDA for unsupported nodes. OWLv2 and BoxerNetCore stay on `CUDAExecutionProvider`.

### Why `onnxruntime-gpu` from `pypi.jetson-ai-lab.io`?

The standard `onnxruntime` package on PyPI is CPU-only and reports `CUDAExecutionProvider` as unavailable on Jetson. The Jetson-specific GPU build with CUDA and TensorRT support is published at:

```
https://pypi.jetson-ai-lab.io/jp6/cu128/
```

This wheel enables both `CUDAExecutionProvider` and `TensorrtExecutionProvider`.

### Why `numpy<2`?

PyTorch 2.7 in `dustynv/pytorch:2.7-r36.4.0` was compiled against NumPy 1.x. NumPy 2.x is binary-incompatible and causes `RuntimeError: Numpy is not available` when importing PyTorch. The base image ships NumPy 2.x, so `numpy<2` is pinned explicitly in both Docker images.

### Why DinoV3 ONNX has no height/width dynamic axes?

The initial export used `{0:"batch", 2:"height", 3:"width"}` dynamic axes for DinoV3. This caused the ONNX exporter to produce symbolic-shape-dependent `If` nodes for the rope position embedding, where each branch has a different output shape. TRT cannot parse `If` nodes with mismatched branch output shapes. Removing height and width dynamic axes eliminates this problem. The `fix_dinov3_onnx.py` script additionally inlines any remaining `If` nodes as a safety measure.

---

## Simple dataset format

A minimal per-frame format. Any directory containing a `meta.json` is auto-detected by `run_boxer.py` as simple format.

### Directory layout

```
<data_dir>/
  meta.json
  frames/
    00000_image.jpg            # or .png
    00000_points.npy           # (N, 3) float32 semi-dense world-coord points
    00000_pose.json            # T_world_camera (4x4)
    00001_image.jpg
    00001_points.npy
    00001_pose.json
    ...
```

Frame filename prefixes (`00000`, `00001`, …) are loaded in sorted order. A zero-padded 5-digit numeric tag is recommended but any alphanumeric string works (used verbatim as the frame tag).

### `meta.json`

Camera intrinsics shared across the sequence.

```json
{
  "camera": {
    "width": 1408,
    "height": 1408,
    "type": "pinhole",
    "fx": 750.0,
    "fy": 750.0,
    "cx": 704.0,
    "cy": 704.0
  },
  "rotated": true,
  "device_name": "my_robot",
  "camera_name": "rgb"
}
```

| Key | Required | Description |
|---|---|---|
| `camera.width`, `camera.height` | ✅ | Image size (px). |
| `camera.type` | ✅ | Currently only `"pinhole"` is supported. |
| `camera.fx`, `camera.fy`, `camera.cx`, `camera.cy` | ✅ | Pinhole intrinsics. |
| `rotated` | ❌ (default `false`) | Set to `true` when stored images are 90° rotated from human-upright. See caveat below. |
| `device_name` | ❌ | Free-form label shown in visualizations. |
| `camera_name` | ❌ | Free-form label shown in visualizations. |

### `NNNNN_image.(jpg|png)`

RGB image. Store in whatever orientation matches the `rotated` flag in `meta.json`. The loader does no re-rotation; the flag is forwarded to BoxerNet, which performs its own internal rotation handling.

### `NNNNN_points.npy`

`(N, 3)` `float32` NumPy array of **3D points in world coordinates**.

- N is per-frame and may vary. Aria typically yields a few thousand points; CA-1M-style depth-map subsampling can produce ~100k.
- Source is unconstrained: depth-map unprojection, LiDAR scans, SLAM semi-dense reconstructions, etc.
- Do **not** include NaNs — `sdp_to_patches` median computation will misbehave.

### `NNNNN_pose.json`

Camera pose in world coordinates as a 4×4 matrix.

```json
{
  "T_world_camera": [
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0.0, 0.0, 0.0,  1.0]
  ],
  "time_ns": 1700000000000000000
}
```

- Convention: maps a point from camera frame to world frame (`p_world = T_world_camera @ p_camera`).
- `time_ns` is optional. Resolution priority:
  1. `pose.json["time_ns"]` if present.
  2. `int(NNNNN)` if it parses as ≥ 1e12 ns.
  3. Otherwise `index × 1e8` (= 10 FPS pseudo-timestamps).

### Caveat — camera orientation

The shipped sample uses `rotated: true` because it was sourced from an Aria sequence where images are stored 90° rotated from human-upright. BoxerNet's [`gravity_align_T_world_cam`](boxer/utils/gravity.py) was trained with this convention (camera's X axis aligns with gravity).

---

## Repository structure

```
.
├── docker/
│   ├── Dockerfile.convert.jetson   # ONNX export
│   └── Dockerfile.infer.jetson     # Multi-stage inference runtime
├── python/
│   ├── onnx_export.py              # Export OWLv2, DinoV3, BoxerNetCore to ONNX
│   ├── fix_dinov3_onnx.py          # Inline rope_embed If nodes for TRT compatibility
│   └── run_boxer_onnx.py           # ONNX Runtime inference (monkey-patches run_boxer.py)
├── input/                          # Sample dataset in simple format (10 frames)
├── export_onnx.sh                  # Export + auto-fix DinoV3
├── run_onnx.sh                     # Run ONNX Runtime inference
└── benchmark.sh                    # Run inference with tegrastats GPU monitoring
```

---

## License

MIT. The Boxer model is licensed under CC-BY-NC; see the [Boxer repository](https://github.com/facebookresearch/boxer) for details.
