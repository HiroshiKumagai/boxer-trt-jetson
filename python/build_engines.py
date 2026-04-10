#!/usr/bin/env python3
"""
Build TensorRT engines from ONNX models.

Usage:
    python /workspace/python/build_engines.py [--all] [--owl] [--dino] [--boxernet] [--fp16]

Output directory: /workspace/trt_engines/
    owlv2_fp16.plan         - OWLv2 vision detector
    dinov3_fp16.plan        - DinoV3 feature extractor
    boxernet_core_fp16.plan - BoxerNet core

Note: TRT engines are platform-specific (x86_64 and aarch64 engines are not interchangeable).
      Rebuild engines when migrating to a different platform (e.g. Jetson).
"""
import sys
import os
import argparse

ONNX_DIR = "/workspace/onnx_weights"
ENGINE_DIR = "/workspace/trt_engines"


def get_trt_logger():
    import tensorrt as trt
    return trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, fp16=True, dynamic_shapes=None):
    """Build and serialize a TensorRT engine from an ONNX file.

    Args:
        onnx_path: Path to input ONNX file.
        engine_path: Path to output engine file (.plan).
        fp16: Enable FP16 precision.
        dynamic_shapes: Dict of {tensor_name: (min_shape, opt_shape, max_shape)}.
    """
    import tensorrt as trt

    TRT_LOGGER = get_trt_logger()
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"  Parsing ONNX: {os.path.basename(onnx_path)}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  Precision: FP16")
    else:
        print("  Precision: FP32")

    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
            print(f"  Dynamic axis {name}: min={min_shape} opt={opt_shape} max={max_shape}")
        config.add_optimization_profile(profile)

    print("  Building engine... (this may take several minutes)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed.")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"  Saved: {engine_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Per-model build functions
# ---------------------------------------------------------------------------

def build_owlv2(fp16, onnx_dir, engine_dir):
    print("==> Building OWLv2 TRT engine...")
    onnx_path = os.path.join(onnx_dir, "owlv2_vision_detector.onnx")
    suffix = "fp16" if fp16 else "fp32"
    engine_path = os.path.join(engine_dir, f"owlv2_{suffix}.plan")

    # text_embeddings and query_mask are fixed at 1220 prompts
    # pixel_values is fixed at 1x3x960x960 (batch=1 only)
    dynamic_shapes = {
        "pixel_values": (
            (1, 3, 960, 960),  # min
            (1, 3, 960, 960),  # opt
            (1, 3, 960, 960),  # max
        ),
    }
    build_engine(onnx_path, engine_path, fp16=fp16, dynamic_shapes=dynamic_shapes)
    return engine_path


def build_dinov3(fp16, onnx_dir, engine_dir):
    print("==> Building DinoV3 TRT engine...")
    onnx_path = os.path.join(onnx_dir, "dinov3.onnx")
    suffix = "fp16" if fp16 else "fp32"
    engine_path = os.path.join(engine_dir, f"dinov3_{suffix}.plan")

    # Fixed at 960x960 (divisible by patch_size=16)
    dynamic_shapes = {
        "img": (
            (1, 3, 960, 960),  # min
            (1, 3, 960, 960),  # opt
            (1, 3, 960, 960),  # max
        ),
    }
    build_engine(onnx_path, engine_path, fp16=fp16, dynamic_shapes=dynamic_shapes)
    return engine_path


def build_boxernet_core(fp16, onnx_dir, engine_dir):
    print("==> Building BoxerNetCore TRT engine...")
    onnx_path = os.path.join(onnx_dir, "boxernet_core.onnx")
    suffix = "fp16" if fp16 else "fp32"
    engine_path = os.path.join(engine_dir, f"boxernet_core_{suffix}.plan")

    # num_patches = 3600 (960/16 * 960/16) is fixed
    # num_boxes is dynamic (1 to 200)
    dynamic_shapes = {
        "x_combined": (
            (1, 3600, 391),  # min
            (1, 3600, 391),  # opt
            (1, 3600, 391),  # max (num_patches is always fixed)
        ),
        "bb2d_norm": (
            (1,   1, 4),  # min
            (1,  20, 4),  # opt (typical detection count)
            (1, 200, 4),  # max
        ),
    }
    build_engine(onnx_path, engine_path, fp16=fp16, dynamic_shapes=dynamic_shapes)
    return engine_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX models")
    parser.add_argument("--onnx_dir", default=ONNX_DIR, help="Input ONNX directory")
    parser.add_argument("--engine_dir", default=ENGINE_DIR, help="Output engine directory")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision (default: enabled)")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision")
    parser.add_argument("--all", action="store_true", help="Build all models")
    parser.add_argument("--owl", action="store_true", help="Build OWLv2 engine")
    parser.add_argument("--dino", action="store_true", help="Build DinoV3 engine")
    parser.add_argument("--boxernet", action="store_true", help="Build BoxerNetCore engine")
    args = parser.parse_args()

    if args.all:
        args.owl = args.dino = args.boxernet = True

    if not (args.owl or args.dino or args.boxernet):
        print("Please specify a target: --all, --owl, --dino, --boxernet")
        parser.print_help()
        sys.exit(1)

    fp16 = not args.fp32

    os.makedirs(args.engine_dir, exist_ok=True)
    built = []

    if args.owl:
        built.append(build_owlv2(fp16, args.onnx_dir, args.engine_dir))

    if args.dino:
        built.append(build_dinov3(fp16, args.onnx_dir, args.engine_dir))

    if args.boxernet:
        built.append(build_boxernet_core(fp16, args.onnx_dir, args.engine_dir))

    print("\n==> Build complete:")
    for p in built:
        print(f"  {p}")


if __name__ == "__main__":
    main()
