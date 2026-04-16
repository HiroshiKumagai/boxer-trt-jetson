#!/usr/bin/env python3
"""
Run Boxer inference using TensorRT engines.

Usage:
    python /workspace/python/run_boxer_trt.py --input sample_data/hohen_gen1 --track

Accepts all the same arguments as run_boxer.py.
OWLv2, DinoV3, and BoxerNetCore are replaced with TensorRT sessions via monkey-patching.
Run build_engines.py first to generate the TRT engine files.
"""
import sys
import os

sys.path.insert(0, "/workspace/boxer")
sys.path.insert(0, "/workspace/python")

ENGINE_DIR = "/workspace/trt_engines"

import numpy as np
import torch
import torch.nn.functional as F


# ===========================================================================
# TRTSession
# Wraps a single TensorRT engine for inference.
# Uses PyTorch tensors as GPU buffers (no pycuda dependency).
# ===========================================================================

class TRTSession:
    """TensorRT inference session for a single engine.

    Inputs:  dict of {tensor_name: np.ndarray}
    Outputs: list of np.ndarray in output tensor order
    """

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._device = device

        # Collect input and output tensor names
        self._input_names = []
        self._output_names = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode.name == "INPUT":
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def run(self, inputs_dict):
        """Execute inference.

        Args:
            inputs_dict: {tensor_name: np.ndarray}
        Returns:
            List of np.ndarray outputs in output_names order.
        """
        import tensorrt as trt

        gpu_buffers = {}

        # Upload inputs to GPU
        for name in self._input_names:
            arr = inputs_dict[name]
            t = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
            gpu_buffers[name] = t
            self._context.set_input_shape(name, arr.shape)
            self._context.set_tensor_address(name, t.data_ptr())

        # Allocate output buffers
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            t = torch.empty(shape, dtype=torch.float32, device="cuda")
            gpu_buffers[name] = t
            self._context.set_tensor_address(name, t.data_ptr())

        # Run inference
        stream = torch.cuda.current_stream()
        self._context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()

        return [gpu_buffers[name].cpu().numpy() for name in self._output_names]


# ===========================================================================
# OwlWrapperTRT
# Replaces the OWLv2 vision detector with a TensorRT session.
# Provides the same API as OwlWrapper.
# ===========================================================================

class OwlWrapperTRT:
    """OWLv2 TensorRT wrapper. Drop-in replacement for OwlWrapper."""

    def __init__(
        self,
        device="cuda",
        text_prompts=None,
        min_confidence=0.2,
        precision=None,
        warmup=True,
        nms_iou_threshold=0.5,
        engine_dir=ENGINE_DIR,
    ):
        import io
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

        checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        self.image_mean = torch.tensor(config["image_mean"]).view(1, 3, 1, 1)
        self.image_std = torch.tensor(config["image_std"]).view(1, 3, 1, 1)
        self.native_size = tuple(config["image_size"])

        # Load text embeddings from ONNX meta file, or recompute
        meta_path = os.path.normpath(
            os.path.join(engine_dir, "..", "onnx_weights", "owlv2_meta.pt")
        )
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            if meta.get("prompts") == text_prompts:
                self._text_embeddings_np = meta["text_embeddings"].numpy().astype(np.float32)
                self._query_mask_np = meta["query_mask"].numpy()
            else:
                self._text_embeddings_np, self._query_mask_np = self._encode_text(
                    checkpoint, config, text_prompts
                )
        else:
            self._text_embeddings_np, self._query_mask_np = self._encode_text(
                checkpoint, config, text_prompts
            )

        # Load TRT engine
        engine_path = os.path.join(engine_dir, "owlv2_fp32.plan")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f"TRT engine not found: {engine_path}\n"
                "Please run build_engines.py --owl first."
            )
        self._session = TRTSession(engine_path, device)

        print(f"Loaded OWLv2 TRT on {device} with {len(text_prompts)} text prompts")

        if warmup:
            self._warmup()

    def _encode_text(self, checkpoint, config, text_prompts):
        import io
        from owl.clip_tokenizer import CLIPTokenizer
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
        return embeddings.numpy(), np.ones(len(text_prompts), dtype=bool)

    def _warmup(self):
        H, W = self.native_size
        dummy = np.zeros((1, 3, H, W), dtype=np.float32)
        self._session.run({
            "pixel_values": dummy,
            "text_embeddings": self._text_embeddings_np,
            "query_mask": self._query_mask_np,
        })

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
            input_image, size=self.native_size, mode="bicubic", align_corners=False
        )
        pixel_values = pixel_values / 255.0
        pixel_values = (pixel_values - self.image_mean) / self.image_std

        # TRT inference
        outputs = self._session.run({
            "pixel_values": pixel_values.cpu().float().numpy(),
            "text_embeddings": self._text_embeddings_np,
            "query_mask": self._query_mask_np,
        })
        logits = torch.from_numpy(outputs[0]).float()     # [1, patches, num_prompts]
        pred_boxes = torch.from_numpy(outputs[1]).float() # [1, patches, 4]

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
        boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]

        if len(boxes) == 0:
            return empty_return

        if self.nms_iou_threshold < 1.0:
            keep = self._per_class_nms(boxes, scores, labels, self.nms_iou_threshold)
            boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]

        if len(boxes) == 0:
            return empty_return

        # Convert x1,y1,x2,y2 -> x1,x2,y1,y2 convention
        boxes = boxes[:, [0, 2, 1, 3]]

        if rotated:
            x1, x2, y1, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = torch.stack([y1, y2, WW - x2, WW - x1], dim=-1)

        return boxes, scores, labels, None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ===========================================================================
# BoxerNetTRT
# Replaces DinoV3 and BoxerNetCore with TensorRT sessions.
# Provides the same API as BoxerNet.
# ===========================================================================

class BoxerNetTRT:
    """BoxerNet TensorRT wrapper. Drop-in replacement for BoxerNet."""

    # Stores the original load_from_checkpoint before monkey-patching
    _orig_load_from_checkpoint = None

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, device="cuda", engine_dir=ENGINE_DIR):
        # Load original PyTorch model for geometry processing and head
        model = cls._orig_load_from_checkpoint(ckpt_path, device=device)

        dino_path = os.path.join(engine_dir, "dinov3_fp32.plan")
        core_path = os.path.join(engine_dir, "boxernet_core_fp32.plan")

        for path in [dino_path, core_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"TRT engine not found: {path}\n"
                    "Please run build_engines.py --dino --boxernet first."
                )

        obj = cls.__new__(cls)
        obj._original_model = model
        obj._device = device
        obj._dino_session = TRTSession(dino_path, device)
        obj._core_session = TRTSession(core_path, device)

        print(f"Loaded BoxerNetTRT on {device} (DinoV3 + Core replaced with TensorRT)")
        return obj

    def __getattr__(self, name):
        # Delegate unknown attribute lookups to the original PyTorch model
        return getattr(self._original_model, name)

    def parameters(self):
        return self._original_model.parameters()

    def named_children(self):
        return self._original_model.named_children()

    def _run_dino_trt(self, torch_img, rotated):
        """Run DinoV3 via TensorRT. Mirrors batch_dino() logic."""
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

        outputs = self._dino_session.run({"img": img.cpu().float().numpy()})
        feat = torch.from_numpy(outputs[0]).to(self._device)

        if pad_h > 0 or pad_w > 0:
            fH = H // patch_size
            fW = W // patch_size
            feat = feat[:, :, :fH, :fW]

        if any_rotated:
            rotated_feat = torch.rot90(feat, 3, [-1, -2])
            feat = torch.where(rotated.reshape(B, 1, 1, 1), rotated_feat, feat)

        return feat

    def _run_core_trt(self, x_combined, bb2d_norm):
        """Run BoxerNetCore via TensorRT. Returns query tokens."""
        outputs = self._core_session.run({
            "x_combined": x_combined.cpu().float().numpy(),
            "bb2d_norm": bb2d_norm.cpu().float().numpy(),
        })
        return torch.from_numpy(outputs[0]).to(self._device)

    @torch.no_grad()
    def forward(self, datum):
        from boxernet.boxernet import (
            sdp_to_patches,
            gravity_align_T_world_cam,
            generate_plucker_encoding,
        )

        inputs = self._original_model.prepare_inputs(datum)

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

        # DinoV3 via TRT: [B, 384, fH, fW]
        dino_feat = self._run_dino_trt(torch_img, rotated)
        _, _, fH, fW = dino_feat.shape
        dino_feat_flat = dino_feat.reshape(B, -1, fH * fW).permute(0, 2, 1)

        # SDP -> per-patch depth
        sdp_w = inputs["sdp_w_padded"]
        sdp_median = sdp_to_patches(sdp_w, cam, T_wr, H, W, self._original_model.dino.patch_size)
        sdp_input = sdp_median.reshape(B, -1, fH * fW).permute(0, 2, 1)

        x_combined = torch.cat([dino_feat_flat, sdp_input], dim=-1)

        # Append Plucker ray encoding if with_ray=True
        if self._original_model.with_ray:
            T_vc = T_wv.inverse() @ T_wc
            ray_enc = generate_plucker_encoding(
                B, fH, fW, self._original_model.dino.patch_size, cam, T_vc
            )
            x_combined = torch.cat([x_combined, ray_enc], dim=-1)

        # Normalize 2D boxes to [0, 1]
        bb2d = inputs["bb2d"]
        M = bb2d.shape[1]
        bb2d_norm = bb2d.clone().float()
        bb2d_norm[..., :2] = (bb2d_norm[..., :2] + 0.5) / W
        bb2d_norm[..., 2:] = (bb2d_norm[..., 2:] + 0.5) / H

        # BoxerNetCore via TRT -> query tokens
        query = self._run_core_trt(x_combined, bb2d_norm)

        # Run head in PyTorch (includes ObbTW construction)
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
    owl_module.OwlWrapper = OwlWrapperTRT

    # Patch BoxerNet — save original before patching to avoid infinite recursion
    import boxernet.boxernet as bn_module
    BoxerNetTRT._orig_load_from_checkpoint = bn_module.BoxerNet.load_from_checkpoint

    @classmethod
    def _patched_load(cls, ckpt_path, device="cuda"):
        return BoxerNetTRT.load_from_checkpoint(ckpt_path, device=device)

    bn_module.BoxerNet.load_from_checkpoint = _patched_load

    import run_boxer
    run_boxer.main()


if __name__ == "__main__":
    _patch_and_run()
