#!/usr/bin/env python3
"""
Run Boxer inference using ONNX Runtime.

Usage:
    python /workspace/python/run_boxer_onnx.py --input sample_data/hohen_gen1 --track

Accepts all the same arguments as run_boxer.py.
OWLv2, DinoV3, and BoxerNetCore are replaced with ONNX Runtime sessions via monkey-patching.
Run onnx_export.py first to generate the ONNX models.
"""
import sys
import os
import io

sys.path.insert(0, "/workspace/boxer")
sys.path.insert(0, "/workspace/python")

ONNX_DIR = "/workspace/onnx_weights"

import numpy as np
import torch
import torch.nn.functional as F


# ===========================================================================
# OwlWrapperONNX
# Replaces the OWLv2 vision detector with an ONNX Runtime session.
# Provides the same API as OwlWrapper.
# ===========================================================================

class OwlWrapperONNX:
    """OWLv2 ONNX Runtime wrapper. Drop-in replacement for OwlWrapper."""

    def __init__(
        self,
        device="cuda",
        text_prompts=None,
        min_confidence=0.2,
        precision=None,
        warmup=True,
        nms_iou_threshold=0.5,
        onnx_dir=ONNX_DIR,
    ):
        import onnxruntime as ort
        from owl.owl_wrapper import _CKPT_PATH, DEFAULT_TEXT_LABELS
        from owl.clip_tokenizer import CLIPTokenizer
        from owl.owl_wrapper import _per_class_nms

        self._per_class_nms = _per_class_nms
        self.device = device
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold

        if text_prompts is None:
            text_prompts = DEFAULT_TEXT_LABELS
        self.text_prompts = text_prompts

        # Load preprocessing config from checkpoint
        checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        self.image_mean = torch.tensor(config["image_mean"]).view(1, 3, 1, 1)
        self.image_std = torch.tensor(config["image_std"]).view(1, 3, 1, 1)
        self.native_size = tuple(config["image_size"])

        # Load or recompute text embeddings
        meta_path = os.path.join(onnx_dir, "owlv2_meta.pt")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"ONNX meta file not found: {meta_path}\n"
                "Please run onnx_export.py --owl first."
            )

        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

        if meta.get("prompts") == text_prompts:
            self._text_embeddings_np = meta["text_embeddings"].numpy().astype(np.float32)
            self._query_mask_np = meta["query_mask"].numpy()
        else:
            print("  Text prompts changed. Recomputing embeddings...")
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
                embeddings = text_encoder(
                    tokens["input_ids"], tokens["attention_mask"]
                ).float()
            self._text_embeddings_np = embeddings.numpy()
            self._query_mask_np = np.ones(len(text_prompts), dtype=bool)

        # Create ONNX Runtime session
        onnx_path = os.path.join(onnx_dir, "owlv2_vision_detector.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Please run onnx_export.py --owl first."
            )

        # OWLv2 の Conv ノードは TensorrtExecutionProvider 非対応のため CUDA のみ使用
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)

        print(f"Loaded OWLv2 ONNX on {device} with {len(text_prompts)} text prompts")

        if warmup:
            self._warmup()

    def _warmup(self):
        H, W = self.native_size
        dummy = np.zeros((1, 3, H, W), dtype=np.float32)
        self._session.run(
            None,
            {
                "pixel_values": dummy,
                "text_embeddings": self._text_embeddings_np,
                "query_mask": self._query_mask_np,
            },
        )

    def set_text_prompts(self, prompts):
        raise NotImplementedError("Dynamic prompt switching is not supported in ONNX mode.")

    @torch.no_grad()
    def forward(self, image_torch, rotated=False, resize_to_HW=(906, 906)):
        assert len(image_torch.shape) == 4
        assert image_torch.shape[0] == 1

        input_image = image_torch.clone()
        if rotated:
            input_image = torch.rot90(input_image, k=3, dims=(2, 3))
        HH, WW = input_image.shape[2], input_image.shape[3]

        # Preprocess
        pixel_values = F.interpolate(
            input_image,
            size=self.native_size,
            mode="bicubic",
            align_corners=False,
        )
        pixel_values = pixel_values / 255.0
        mean = self.image_mean.to(pixel_values.device)
        std = self.image_std.to(pixel_values.device)
        pixel_values = (pixel_values - mean) / std

        # ONNX inference
        pixel_values_np = pixel_values.cpu().float().numpy()
        outputs = self._session.run(
            None,
            {
                "pixel_values": pixel_values_np,
                "text_embeddings": self._text_embeddings_np,
                "query_mask": self._query_mask_np,
            },
        )
        logits = torch.from_numpy(outputs[0]).float()    # [1, num_patches, num_queries]
        pred_boxes = torch.from_numpy(outputs[1]).float() # [1, num_patches, 4]

        # Postprocess (identical to OwlWrapper.forward())
        scores_all, labels_all = torch.max(logits[0], dim=-1)
        scores_all = torch.sigmoid(scores_all)

        keep = scores_all > self.min_confidence
        scores = scores_all[keep].cpu()
        labels = labels_all[keep].cpu()
        boxes_cxcywh = pred_boxes[0, keep]

        empty_return = torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0), None
        if len(boxes_cxcywh) == 0:
            return empty_return

        cx, cy, w, h = boxes_cxcywh.unbind(-1)
        x1 = (cx - w / 2) * WW
        y1 = (cy - h / 2) * HH
        x2 = (cx + w / 2) * WW
        y2 = (cy + h / 2) * HH
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu()

        too_big = (x2 - x1 > 0.9 * WW) | (y2 - y1 > 0.9 * HH)
        too_small = (x2 - x1 < 0.05 * WW) | (y2 - y1 < 0.05 * HH)
        keep = ~(too_big | too_small).cpu()
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            return empty_return

        if self.nms_iou_threshold < 1.0:
            keep = self._per_class_nms(boxes, scores, labels, self.nms_iou_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        if len(boxes) == 0:
            return empty_return

        # Convert x1,y1,x2,y2 -> x1,x2,y1,y2 convention
        boxes = boxes[:, [0, 2, 1, 3]]

        if rotated:
            x1, x2, y1, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            new_x1 = y1
            new_x2 = y2
            new_y1 = WW - x2
            new_y2 = WW - x1
            boxes = torch.stack([new_x1, new_x2, new_y1, new_y2], dim=-1)

        return boxes, scores, labels, None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ===========================================================================
# BoxerNetONNX
# Replaces DinoV3 and BoxerNetCore with ONNX Runtime sessions.
# Provides the same API as BoxerNet.
# ===========================================================================

class BoxerNetONNX:
    """BoxerNet ONNX Runtime wrapper. Drop-in replacement for BoxerNet."""

    # Stores the original load_from_checkpoint before monkey-patching
    _orig_load_from_checkpoint = None

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, device="cuda", onnx_dir=ONNX_DIR):
        import onnxruntime as ort

        # Load original PyTorch model for geometry processing and head
        model = cls._orig_load_from_checkpoint(ckpt_path, device=device)

        cuda_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        # DinoV3 は固定形状 (1,3,960,960) → TensorrtExecutionProvider で高速化
        # BoxerNetCore は動的 M 次元のため TRT 非対応 → CUDA のまま
        trt_providers = (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        dino_path = os.path.join(onnx_dir, "dinov3.onnx")
        core_path = os.path.join(onnx_dir, "boxernet_core.onnx")

        for path in [dino_path, core_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"ONNX model not found: {path}\n"
                    "Please run onnx_export.py --dino --boxernet first."
                )

        obj = cls.__new__(cls)
        obj.__dict__.update(model.__dict__)
        obj._original_model = model
        obj._device = device

        # DinoV3: TensorrtExecutionProvider (fixed shape 1×3×960×960, cache reused across runs)
        # Set trt_fp16_enable=True for ~14% faster inference at ~7% fewer 3D detections.
        # Set trt_fp16_enable=False (or remove the key) for full FP32 accuracy.
        trt_cache = os.environ.get('ORT_TENSORRT_CACHE_PATH', '/workspace/trt_cache')
        dino_trt_providers = (
            [('TensorrtExecutionProvider', {
                'trt_fp16_enable': False,   # set True for FP16 (faster, slightly less accurate)
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': trt_cache,
            }), 'CUDAExecutionProvider', 'CPUExecutionProvider']
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        # BoxerNetCore: CUDAExecutionProvider only.
        # The M dimension (number of 2D detections) is dynamic and changes every frame,
        # which causes TensorrtExecutionProvider to recompile the engine per unique M value (~27 s each).
        obj._dino_session = ort.InferenceSession(dino_path, sess_opts, providers=dino_trt_providers)
        obj._core_session = ort.InferenceSession(core_path, sess_opts, providers=cuda_providers)

        print(f"Loaded BoxerNetONNX on {device} (DinoV3 + Core replaced with ONNX Runtime)")
        return obj

    # -- nn.Module compatibility stubs ------------------------------------

    def __getattr__(self, name):
        return getattr(self._original_model, name)

    def parameters(self):
        return self._original_model.parameters()

    def named_children(self):
        return self._original_model.named_children()

    # -- ONNX inference helpers -------------------------------------------

    def _run_dino_onnx(self, torch_img, rotated):
        """Run DinoV3 via ONNX Runtime. Mirrors batch_dino() logic."""
        B = torch_img.shape[0]
        patch_size = self._original_model.dino.patch_size
        any_rotated = rotated.any().item()

        img = torch_img.clone()
        if any_rotated:
            rotated_img = torch.rot90(img, 1, [-1, -2])
            img = torch.where(rotated.reshape(B, 1, 1, 1), rotated_img, img)

        _, _, H, W = img.shape
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h))

        img_np = img.cpu().float().numpy()
        outputs = self._dino_session.run(None, {"img": img_np})
        feat = torch.from_numpy(outputs[0]).to(self._device)

        if pad_h > 0 or pad_w > 0:
            fH = H // patch_size
            fW = W // patch_size
            feat = feat[:, :, :fH, :fW]

        if any_rotated:
            rotated_feat = torch.rot90(feat, 3, [-1, -2])
            feat = torch.where(rotated.reshape(B, 1, 1, 1), rotated_feat, feat)

        return feat

    def _run_core_onnx(self, x_combined, bb2d_norm):
        """Run BoxerNetCore via ONNX Runtime. Returns query tokens."""
        x_np = x_combined.cpu().float().numpy()
        b_np = bb2d_norm.cpu().float().numpy()
        outputs = self._core_session.run(
            None, {"x_combined": x_np, "bb2d_norm": b_np}
        )
        return torch.from_numpy(outputs[0]).to(self._device)

    # -- forward ----------------------------------------------------------

    @torch.no_grad()
    def forward(self, datum):
        from boxernet.boxernet import (
            sdp_to_patches,
            gravity_align_T_world_cam,
            generate_plucker_encoding,
        )

        inputs = self._original_model.prepare_inputs(datum)

        # --- Equivalent to encode() ---
        torch_img = inputs["img0"]
        cam = inputs["cam0"]
        T_wr = inputs["T_world_rig0"]
        T_wc = T_wr @ cam.T_camera_rig.inverse()

        if "T_world_voxel0" not in inputs:
            T_wv = gravity_align_T_world_cam(T_wc, z_grav=True)
            inputs["T_world_voxel0"] = T_wv
        T_wv = inputs["T_world_voxel0"]

        rotated = inputs["rotated0"]
        B, _, H, W = torch_img.shape

        # DinoV3 via ONNX: [B, 384, fH, fW]
        dino_feat = self._run_dino_onnx(torch_img, rotated)
        _, _, fH, fW = dino_feat.shape
        dino_feat_flat = dino_feat.reshape(B, -1, fH * fW).permute(0, 2, 1)  # [B, fH*fW, 384]

        # SDP -> per-patch depth
        sdp_w = inputs["sdp_w_padded"]
        sdp_median = sdp_to_patches(
            sdp_w, cam, T_wr, H, W, self._original_model.dino.patch_size
        )
        sdp_input = sdp_median.reshape(B, -1, fH * fW).permute(0, 2, 1)  # [B, fH*fW, 1]

        x_combined = torch.cat([dino_feat_flat, sdp_input], dim=-1)  # [B, fH*fW, 385]

        # Append Plucker ray encoding if with_ray=True
        if self._original_model.with_ray:
            T_vc = T_wv.inverse() @ T_wc
            ray_enc = generate_plucker_encoding(
                B, fH, fW, self._original_model.dino.patch_size, cam, T_vc
            )  # [B, fH*fW, 6]
            x_combined = torch.cat([x_combined, ray_enc], dim=-1)  # [B, fH*fW, 391]

        # --- Equivalent to query() ---
        bb2d = inputs["bb2d"]
        M = bb2d.shape[1]

        # Normalize 2D boxes to [0, 1]
        bb2d_norm = bb2d.clone().float()
        bb2d_norm[..., :2] = (bb2d_norm[..., :2] + 0.5) / W
        bb2d_norm[..., 2:] = (bb2d_norm[..., 2:] + 0.5) / H

        # BoxerNetCore via ONNX -> query tokens: [B, M, dim]
        query = self._run_core_onnx(x_combined, bb2d_norm)

        # --- Run head in PyTorch (includes ObbTW construction) ---
        if "obbs_valid" not in inputs:
            inputs["obbs_valid"] = torch.ones(
                (B, M, 1), dtype=torch.bool, device=self._device
            )
        output = {"query": query, "dino0": dino_feat}
        inputs, output = self._original_model.head(inputs, output)

        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ===========================================================================
# Monkey-patch and invoke run_boxer.main()
# ===========================================================================

def _patch_and_run():
    # Patch OWLv2
    import owl.owl_wrapper as owl_module

    class _OwlProxy(OwlWrapperONNX):
        pass

    owl_module.OwlWrapper = _OwlProxy

    # Patch BoxerNet — save original before patching to avoid infinite recursion
    import boxernet.boxernet as bn_module
    BoxerNetONNX._orig_load_from_checkpoint = bn_module.BoxerNet.load_from_checkpoint

    @classmethod
    def _patched_load(cls, ckpt_path, device="cuda"):
        return BoxerNetONNX.load_from_checkpoint(ckpt_path, device=device)

    bn_module.BoxerNet.load_from_checkpoint = _patched_load

    import run_boxer
    run_boxer.main()


if __name__ == "__main__":
    _patch_and_run()
