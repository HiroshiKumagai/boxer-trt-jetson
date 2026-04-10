# boxer-trt-x86_64

TensorRT acceleration for [Boxer](https://github.com/facebookresearch/boxer) on x86_64 (CUDA).

This repository exports all three Boxer models (OWLv2, DinoV3, BoxerNetCore) to ONNX and compiles them into TensorRT FP16 engines, achieving approximately **~2× faster inference** compared to the original PyTorch pipeline.

Inference scripts are provided for both ONNX Runtime and TensorRT backends. The original `run_boxer.py` is used unchanged via monkey-patching.

---

## Architecture

```
OWLv2 checkpoint + DinoV3 checkpoint + BoxerNet checkpoint
         ↓  export_onnx.sh
owlv2_vision_detector.onnx / dinov3.onnx / boxernet_core.onnx
         ↓  build_engine.sh
owlv2_fp16.plan / dinov3_fp16.plan / boxernet_core_fp16.plan
         ↓  run_trt.sh
3D bounding boxes + tracking results + visualization video
```

| Component | Backend | Latency (per frame) |
|---|---|---|
| OWLv2 vision detector | PyTorch | ~100 ms |
| OWLv2 vision detector | ONNX Runtime | ~45 ms |
| OWLv2 vision detector | **TensorRT FP16** | **~16 ms** |
| DinoV3 + BoxerNetCore | PyTorch | ~47 ms |
| DinoV3 + BoxerNetCore | ONNX Runtime | ~44 ms |
| DinoV3 + BoxerNetCore | **TensorRT FP16** | **~19 ms** |

*Measured on NVIDIA RTX PRO 6000 Blackwell (sm_120), CUDA 12.8, TensorRT 10.x*

---

## Requirements

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit (`nvidia-docker2` or `--runtime=nvidia`)
- Boxer repository cloned to `boxer/`
- Boxer model checkpoints in `ckpts/`
- Sample data in `sample_data/` (optional, for testing)

---

## Setup

### 1. Clone Boxer

```bash
git clone https://github.com/facebookresearch/boxer.git boxer
```

### 2. Download model checkpoints

```bash
bash boxer/scripts/download_ckpts.sh
mv ckpts boxer/ckpts   # or symlink
```

Alternatively, place checkpoints directly in `ckpts/`:
- `boxernet_hw960in4x6d768-wssxpf9p.ckpt`
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`
- `owlv2-base-patch16-ensemble.pt`

### 3. Download sample data (optional)

```bash
bash boxer/scripts/download_aria_data.sh hohen_gen1
mv hohen_gen1 sample_data/
```

### 4. Build Docker images

```bash
docker build -f docker/Dockerfile.convert.x86_64 -t boxer-convert-x86_64 .
docker build -f docker/Dockerfile.infer.x86_64   -t boxer-infer-x86_64   .
```

---

## Usage

### Step 1 — Export models to ONNX

```bash
bash export_onnx.sh
```

Outputs to `onnx_weights/`:
- `owlv2_vision_detector.onnx` + `owlv2_meta.pt`
- `dinov3.onnx`
- `boxernet_core.onnx`

### Step 2 — Build TensorRT engines

```bash
bash build_engine.sh
```

Outputs to `trt_engines/`:
- `owlv2_fp16.plan`
- `dinov3_fp16.plan`
- `boxernet_core_fp16.plan`

> TRT engines are platform-specific. Rebuild on each target platform.

### Step 3 — Run inference

**ONNX Runtime:**
```bash
bash run_onnx.sh --input sample_data/hohen_gen1 --track
```

**TensorRT:**
```bash
bash run_trt.sh --input sample_data/hohen_gen1 --track
```

Outputs to `output/`:
- `boxer_3dbbs.csv` — per-frame 3D bounding boxes
- `owl_2dbbs.csv` — per-frame 2D bounding boxes
- `boxer_3dbbs_tracked.csv` — tracked 3D bounding boxes
- `boxer_viz/` — per-frame visualization images
- `boxer_viz_final.mp4` — visualization video

---

## Repository Structure

```
boxer-trt-x86_64/
├── docker/
│   ├── Dockerfile.convert.x86_64  # ONNX export + TRT engine build
│   └── Dockerfile.infer.x86_64    # ONNX and TRT inference
├── python/
│   ├── onnx_export.py             # Export models to ONNX
│   ├── build_engines.py           # Build TRT engines from ONNX
│   ├── run_boxer_onnx.py          # ONNX Runtime inference
│   └── run_boxer_trt.py           # TensorRT inference
├── .gitignore
├── export_onnx.sh                 # Wrapper: ONNX export
├── build_engine.sh                # Wrapper: TRT engine build
├── run_onnx.sh                    # Wrapper: ONNX inference
└── run_trt.sh                     # Wrapper: TRT inference
```

---

## Jetson Migration

To run on Jetson (aarch64), the Python scripts require no changes.
Rebuild the Docker images using JetPack-based images (e.g. `dustynv/pytorch`) and re-run `build_engine.sh` on the Jetson device to generate aarch64-compatible TRT engines.

---

## License

MIT License. See [LICENSE](LICENSE).

The Boxer model itself is licensed under CC-BY-NC. See the [Boxer repository](https://github.com/facebookresearch/boxer) for details.
