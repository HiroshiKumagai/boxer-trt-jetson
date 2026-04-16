#!/usr/bin/env python3
"""
Export Boxer models (OWLv2, DinoV3, BoxerNetCore) to ONNX format.

Usage:
    python /workspace/python/onnx_export.py [--ckpt <path>] [--all] [--owl] [--dino] [--boxernet]

Output directory: /workspace/onnx_weights/
    owlv2_vision_detector.onnx  - OWLv2 vision detector
    owlv2_meta.pt               - text embeddings and config
    dinov3.onnx                 - DinoV3 feature extractor
    boxernet_core.onnx          - BoxerNet core (attention layers)
"""
import sys
import os
import argparse
import torch
import numpy as np

sys.path.insert(0, "/workspace/boxer")
sys.path.insert(0, "/workspace/python")

ONNX_DIR = "/workspace/onnx_weights"
DEFAULT_CKPT = "/workspace/boxer/ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt"


# ---------------------------------------------------------------------------
# OWLv2 VisionDetector
# ---------------------------------------------------------------------------

def export_owlv2(device, onnx_dir, text_prompts=None):
    import io
    from owl.owl_wrapper import VisionDetectorWrapper, _CKPT_PATH
    from owl.clip_tokenizer import CLIPTokenizer

    print("==> Exporting OWLv2 VisionDetector to ONNX...")

    if not os.path.exists(_CKPT_PATH):
        raise FileNotFoundError(f"OWLv2 checkpoint not found: {_CKPT_PATH}")

    checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    if text_prompts is None:
        from owl.owl_wrapper import DEFAULT_TEXT_LABELS
        text_prompts = DEFAULT_TEXT_LABELS

    # Compute text embeddings on CPU
    text_encoder = torch.jit.load(
        io.BytesIO(checkpoint["text_encoder"]), map_location="cpu"
    ).eval()
    tokenizer = CLIPTokenizer(
        vocab=checkpoint["tokenizer_vocab"],
        merges=checkpoint["tokenizer_merges"],
        max_length=config["max_seq_length"],
    )
    tokens = tokenizer(text_prompts)
    with torch.no_grad():
        text_embeddings = text_encoder(
            tokens["input_ids"], tokens["attention_mask"]
        ).float()  # [N, 512] float32

    query_mask = torch.ones(len(text_prompts), dtype=torch.bool)

    # Export VisionDetector (uncompiled, on CPU)
    vision_detector = VisionDetectorWrapper.from_state_dict(
        checkpoint["vision_detector_state_dict"]
    ).eval()

    H, W = config["image_size"]
    pixel_values = torch.zeros(1, 3, H, W, dtype=torch.float32)

    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "owlv2_vision_detector.onnx")

    with torch.no_grad():
        torch.onnx.export(
            vision_detector,
            (pixel_values, text_embeddings, query_mask),
            onnx_path,
            input_names=["pixel_values", "text_embeddings", "query_mask"],
            output_names=["pred_logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "pred_logits": {0: "batch"},
                "pred_boxes": {0: "batch"},
            },
            opset_version=18,
            dynamo=False,
        )

    # Save meta information (text embeddings and preprocessing config)
    meta_path = os.path.join(onnx_dir, "owlv2_meta.pt")
    torch.save(
        {
            "text_embeddings": text_embeddings.cpu(),
            "query_mask": query_mask.cpu(),
            "prompts": text_prompts,
            "image_mean": config["image_mean"],
            "image_std": config["image_std"],
            "image_size": config["image_size"],
        },
        meta_path,
    )

    print(f"  Saved: {onnx_path}")
    print(f"  Saved: {meta_path}  ({len(text_prompts)} prompts)")
    return onnx_path


# ---------------------------------------------------------------------------
# DinoV3
# ---------------------------------------------------------------------------

def export_dinov3(device, onnx_dir, hw=960):
    from boxernet.dinov3_wrapper import DinoV3Wrapper

    print("==> Exporting DinoV3 to ONNX...")

    dino = DinoV3Wrapper("dinov3_vits16plus").eval()
    # Export on CPU to avoid CUDA kernel discrepancies
    img = torch.zeros(1, 3, hw, hw, dtype=torch.float32)

    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "dinov3.onnx")

    with torch.no_grad():
        torch.onnx.export(
            dino,
            (img,),
            onnx_path,
            input_names=["img"],
            output_names=["features"],
            # height/width は常に 960x960 固定。動的にすると symbolic shape が
            # rope_embed 内の If ノードを生成し TensorRT がパースできないため固定する。
            dynamic_axes={
                "img": {0: "batch"},
                "features": {0: "batch"},
            },
            opset_version=18,
            dynamo=False,
        )

    print(f"  Saved: {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# BoxerNet Core
# ---------------------------------------------------------------------------

class _BoxerNetCoreONNX(torch.nn.Module):
    """ONNX export wrapper for the BoxerNet attention core.

    Inputs:
        x_combined : [B, N, in_dim]  DinoV3 features concatenated with SDP depth patches
        bb2d_norm  : [B, M, 4]       Normalized 2D bounding boxes in [0, 1]
    Output:
        query      : [B, M, dim]     Query tokens after cross-attention
    """

    def __init__(self, boxernet):
        super().__init__()
        self.input2emb = boxernet.input2emb
        self.query2emb = boxernet.query2emb
        self.in_depth = boxernet.in_depth
        if boxernet.in_depth > 0:
            self.self_attn = boxernet.self_attn
        else:
            self.self_attn = None
        self.cross_attn = boxernet.cross_attn

    def forward(self, x_combined, bb2d_norm):
        input_enc = self.input2emb(x_combined)
        if self.self_attn is not None:
            input_enc = self.self_attn(input_enc)
        query = self.query2emb(bb2d_norm)
        query = self.cross_attn(query, input_enc)
        return query


def export_boxernet_core(device, onnx_dir, ckpt_path):
    from boxernet.boxernet import BoxerNet

    print("==> Exporting BoxerNetCore to ONNX...")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"BoxerNet checkpoint not found: {ckpt_path}")

    # Load and export on CPU
    boxernet = BoxerNet.load_from_checkpoint(ckpt_path, device="cpu")
    core = _BoxerNetCoreONNX(boxernet).eval()

    hw = boxernet.hw
    patch_size = 16
    fH = hw // patch_size
    fW = hw // patch_size
    in_dim = boxernet.in_dim  # 391 = DinoV3(384) + depth(1) + Plucker ray(6)
    dim = boxernet.dim        # 768

    N = fH * fW  # number of patches (e.g. 60*60=3600)
    M = 10       # representative number of 2D boxes (dynamic axis)

    x_combined = torch.zeros(1, N, in_dim, dtype=torch.float32)
    bb2d_norm = torch.zeros(1, M, 4, dtype=torch.float32)

    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "boxernet_core.onnx")

    with torch.no_grad():
        torch.onnx.export(
            core,
            (x_combined, bb2d_norm),
            onnx_path,
            input_names=["x_combined", "bb2d_norm"],
            output_names=["query"],
            dynamic_axes={
                "x_combined": {0: "batch", 1: "num_patches"},
                "bb2d_norm": {0: "batch", 1: "num_boxes"},
                "query": {0: "batch", 1: "num_boxes"},
            },
            opset_version=18,
            dynamo=False,
        )

    print(f"  Saved: {onnx_path}")
    print(f"  in_dim={in_dim}, num_patches={N}, model_dim={dim}")
    return onnx_path


# ---------------------------------------------------------------------------
# Optional: simplify ONNX model with onnxsim
# ---------------------------------------------------------------------------

def simplify_onnx(onnx_path):
    try:
        import onnx
        from onnxsim import simplify

        print(f"  Simplifying with onnxsim: {os.path.basename(onnx_path)} ...")
        model = onnx.load(onnx_path)
        model_simplified, check = simplify(model)
        if check:
            onnx.save(model_simplified, onnx_path)
            print("  Simplification done.")
        else:
            print("  Simplification skipped (verification failed).")
    except ImportError:
        print("  onnxsim not installed, skipping.")
    except Exception as e:
        print(f"  Simplification error (skipping): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export Boxer models to ONNX")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="BoxerNet checkpoint path")
    parser.add_argument("--onnx_dir", default=ONNX_DIR, help="Output directory for ONNX files")
    parser.add_argument("--device", default="cpu", help="Device for export (cpu recommended)")
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument("--owl", action="store_true", help="Export OWLv2")
    parser.add_argument("--dino", action="store_true", help="Export DinoV3")
    parser.add_argument("--boxernet", action="store_true", help="Export BoxerNetCore")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX with onnxsim")
    args = parser.parse_args()

    if args.all:
        args.owl = args.dino = args.boxernet = True

    if not (args.owl or args.dino or args.boxernet):
        print("Please specify a target: --all, --owl, --dino, --boxernet")
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.onnx_dir, exist_ok=True)
    exported = []

    if args.owl:
        path = export_owlv2(args.device, args.onnx_dir)
        if args.simplify:
            simplify_onnx(path)
        exported.append(path)

    if args.dino:
        path = export_dinov3(args.device, args.onnx_dir)
        if args.simplify:
            simplify_onnx(path)
        exported.append(path)

    if args.boxernet:
        path = export_boxernet_core(args.device, args.onnx_dir, args.ckpt)
        if args.simplify:
            simplify_onnx(path)
        exported.append(path)

    print("\n==> Export complete:")
    for p in exported:
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"  {p}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
