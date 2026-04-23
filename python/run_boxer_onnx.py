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


def _cuda_ep_opts(strict_arena: bool = False) -> dict:
    """Return CUDAExecutionProvider options from env vars (Phase 1 RAM limits).

    BOXER_CUDA_MEM_LIMIT_GB  : cap CUDA allocator growth to N GB (0 = no cap)
    strict_arena             : if True, use kSameAsRequested instead of kNextPowerOfTwo
                               (reduces over-allocation; may hurt Conv perf)
    """
    opts = {}
    try:
        mem_limit_gb = float(os.environ.get("BOXER_CUDA_MEM_LIMIT_GB", "0"))
    except ValueError:
        mem_limit_gb = 0
    if mem_limit_gb > 0:
        opts["gpu_mem_limit"] = int(mem_limit_gb * (1 << 30))
    if strict_arena:
        opts["arena_extend_strategy"] = "kSameAsRequested"
    return opts


def _trt_workspace_bytes() -> int:
    """TRT workspace size, overridable via BOXER_TRT_WORKSPACE_MB (default 1024 MB)."""
    try:
        mb = int(os.environ.get("BOXER_TRT_WORKSPACE_MB", "1024"))
    except ValueError:
        mb = 1024
    return mb * (1 << 20)


def _log_mem(tag: str) -> None:
    """Print CUDA + process memory snapshot for Phase 1 profiling."""
    if os.environ.get("BOXER_MEM_PROFILE") != "1":
        return
    torch_alloc_gb = torch.cuda.memory_allocated() / (1 << 30)
    torch_reserved_gb = torch.cuda.memory_reserved() / (1 << 30)
    # RSS from /proc/self/status in KiB
    rss_gb = 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = int(line.split()[1]) / (1 << 20)
                    break
    except Exception:
        pass
    # GPU free/total (on Jetson this is unified memory; reflects system pressure)
    gpu_free_gb = gpu_total_gb = 0.0
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_free_gb = free_b / (1 << 30)
        gpu_total_gb = total_b / (1 << 30)
    except Exception:
        pass
    print(
        f"[mem] {tag}: torch_alloc={torch_alloc_gb:.2f}GB "
        f"torch_reserved={torch_reserved_gb:.2f}GB "
        f"RSS={rss_gb:.2f}GB "
        f"GPU_free={gpu_free_gb:.2f}/{gpu_total_gb:.2f}GB",
        flush=True,
    )


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
        # Place normalization constants on the inference device to keep preprocessing
        # fully on GPU (avoids per-frame CPU→GPU copies).
        torch_device = torch.device(device) if device != "cpu" else torch.device("cpu")
        self.image_mean = torch.tensor(config["image_mean"]).view(1, 3, 1, 1).to(torch_device)
        self.image_std = torch.tensor(config["image_std"]).view(1, 3, 1, 1).to(torch_device)
        self.native_size = tuple(config["image_size"])
        self._torch_device = torch_device

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
        # CUDA Graph を試したが OWLv2 は CPU fallback ノード (Memcpy x10) が残るため不可。
        # 構造的ブロッカーで回避不能 — IOBinding + persistent buffer のみ適用。
        providers = (
            [("CUDAExecutionProvider", _cuda_ep_opts())]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        if device == "cuda":
            providers.append("CPUExecutionProvider")
        sess_opts = ort.SessionOptions()
        # If a pre-optimized graph exists, load it directly and skip optimization.
        # Otherwise run ORT_ENABLE_ALL and serialize the result for next run.
        # Set BOXER_DISABLE_OPTIMIZED_CACHE=1 to bypass this and force ORT_ENABLE_ALL
        # with no serialization (used to measure Stage 5's standalone effect).
        _disable_opt_cache = os.environ.get("BOXER_DISABLE_OPTIMIZED_CACHE") == "1"
        optimized_path = onnx_path + ".optimized.onnx"
        if (not _disable_opt_cache) and os.path.exists(optimized_path):
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            load_path = optimized_path
        else:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if not _disable_opt_cache:
                sess_opts.optimized_model_filepath = optimized_path
            load_path = onnx_path
        self._session = ort.InferenceSession(load_path, sess_opts, providers=providers)
        _log_mem("after OWL session init")

        # Set up IOBinding + persistent buffers for CUDA Graph replay (CUDA only).
        # - text_embeddings / query_mask: fixed, bound once
        # - pixel_values / outputs: persistent buffers, same data_ptr every frame
        # - Graph is captured on first run_with_iobinding call, then replayed
        self._use_io_binding = (device == "cuda")
        # BOXER_DISABLE_PERSISTENT_BUF=1 reverts Stage 4: allocate logits/boxes with
        # torch.empty every frame and rebind, instead of reusing warmup buffers.
        self._disable_persistent_buf = os.environ.get("BOXER_DISABLE_PERSISTENT_BUF") == "1"
        if self._use_io_binding:
            self._io_binding = self._session.io_binding()
            self._text_emb_ortval = ort.OrtValue.ortvalue_from_numpy(
                self._text_embeddings_np, "cuda", 0
            )
            self._query_mask_ortval = ort.OrtValue.ortvalue_from_numpy(
                self._query_mask_np, "cuda", 0
            )
            self._io_binding.bind_ortvalue_input("text_embeddings", self._text_emb_ortval)
            self._io_binding.bind_ortvalue_input("query_mask", self._query_mask_ortval)
            # Persistent output buffers allocated at warmup
            self._logits_buf = None
            self._boxes_buf = None
            self._logits_shape = None
            self._boxes_shape = None

        print(f"Loaded OWLv2 ONNX on {device} with {len(text_prompts)} text prompts")

        if warmup:
            self._warmup()
            _log_mem("after OWL warmup")

    def _warmup(self):
        H, W = self.native_size
        dummy = np.zeros((1, 3, H, W), dtype=np.float32)
        outputs = self._session.run(
            None,
            {
                "pixel_values": dummy,
                "text_embeddings": self._text_embeddings_np,
                "query_mask": self._query_mask_np,
            },
        )
        if self._use_io_binding:
            logits_shape = outputs[0].shape  # (1, num_patches, num_queries)
            boxes_shape = outputs[1].shape   # (1, num_patches, 4)
            self._logits_shape = logits_shape
            self._boxes_shape = boxes_shape
            if not self._disable_persistent_buf:
                # Persistent output buffers: bind once, reuse every frame.
                # Input (pixel_values) is rebound per-frame from the preprocessed tensor
                # pointer — avoids an extra GPU→GPU copy.
                self._logits_buf = torch.empty(
                    logits_shape, dtype=torch.float32, device=self._torch_device
                )
                self._boxes_buf = torch.empty(
                    boxes_shape, dtype=torch.float32, device=self._torch_device
                )
                dev_id = self._logits_buf.device.index or 0
                self._io_binding.bind_output(
                    name="pred_logits",
                    device_type="cuda",
                    device_id=dev_id,
                    element_type=np.float32,
                    shape=tuple(self._logits_buf.shape),
                    buffer_ptr=self._logits_buf.data_ptr(),
                )
                self._io_binding.bind_output(
                    name="pred_boxes",
                    device_type="cuda",
                    device_id=dev_id,
                    element_type=np.float32,
                    shape=tuple(self._boxes_buf.shape),
                    buffer_ptr=self._boxes_buf.data_ptr(),
                )

    def set_text_prompts(self, prompts):
        raise NotImplementedError("Dynamic prompt switching is not supported in ONNX mode.")

    @torch.no_grad()
    def forward(self, image_torch, rotated=False, resize_to_HW=(906, 906)):
        assert len(image_torch.shape) == 4
        assert image_torch.shape[0] == 1

        # Move input to target device early; preprocessing and inference stay on GPU.
        input_image = image_torch.to(self._torch_device, non_blocking=True)
        if rotated:
            input_image = torch.rot90(input_image, k=3, dims=(2, 3))
        HH, WW = input_image.shape[2], input_image.shape[3]

        # Preprocess on GPU (bicubic interpolate, normalize)
        pixel_values = F.interpolate(
            input_image.float(),
            size=self.native_size,
            mode="bicubic",
            align_corners=False,
        )
        pixel_values = pixel_values / 255.0
        pixel_values = (pixel_values - self.image_mean) / self.image_std

        if self._use_io_binding:
            # Rebind pixel_values from the preprocessed tensor's pointer — no extra copy.
            # Outputs land in persistent buffers bound at warmup.
            pixel_values = pixel_values.contiguous()
            self._io_binding.bind_input(
                name="pixel_values",
                device_type="cuda",
                device_id=pixel_values.device.index or 0,
                element_type=np.float32,
                shape=tuple(pixel_values.shape),
                buffer_ptr=pixel_values.data_ptr(),
            )
            if self._disable_persistent_buf:
                # Stage 4 OFF: allocate output buffers per frame (original Stage A pattern).
                dev_id = pixel_values.device.index or 0
                self._logits_buf = torch.empty(
                    self._logits_shape, dtype=torch.float32, device=self._torch_device
                )
                self._boxes_buf = torch.empty(
                    self._boxes_shape, dtype=torch.float32, device=self._torch_device
                )
                self._io_binding.bind_output(
                    name="pred_logits",
                    device_type="cuda",
                    device_id=dev_id,
                    element_type=np.float32,
                    shape=tuple(self._logits_buf.shape),
                    buffer_ptr=self._logits_buf.data_ptr(),
                )
                self._io_binding.bind_output(
                    name="pred_boxes",
                    device_type="cuda",
                    device_id=dev_id,
                    element_type=np.float32,
                    shape=tuple(self._boxes_buf.shape),
                    buffer_ptr=self._boxes_buf.data_ptr(),
                )
            # ORT's CUDAExecutionProvider runs on its own stream — it does NOT
            # implicitly wait for torch's default stream. Without these syncs,
            # ORT reads pixel_values before torch's preprocessing kernels finish
            # (→ NaN logits) and torch postprocess reads _logits_buf before ORT
            # finishes (→ stale logits). Both races manifest as alternating
            # 0-detection frames.
            self._io_binding.synchronize_inputs()
            self._session.run_with_iobinding(self._io_binding)
            self._io_binding.synchronize_outputs()
            logits = self._logits_buf
            pred_boxes = self._boxes_buf
        else:
            # CPU fallback: numpy path
            pixel_values_np = pixel_values.cpu().float().numpy()
            outputs = self._session.run(
                None,
                {
                    "pixel_values": pixel_values_np,
                    "text_embeddings": self._text_embeddings_np,
                    "query_mask": self._query_mask_np,
                },
            )
            logits = torch.from_numpy(outputs[0]).float()
            pred_boxes = torch.from_numpy(outputs[1]).float()

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

        # Core (CUDAExecutionProvider): apply gpu_mem_limit + optional strict arena.
        # Strict arena reduces over-allocation for dynamic-M input but may hurt perf.
        _core_strict_arena = os.environ.get("BOXER_CORE_ARENA_STRICT") == "1"
        cuda_providers = (
            [("CUDAExecutionProvider", _cuda_ep_opts(strict_arena=_core_strict_arena)),
             "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        _disable_opt_cache = os.environ.get("BOXER_DISABLE_OPTIMIZED_CACHE") == "1"
        def _make_sess_opts_and_path(onnx_file_path, serialize_optimized=True):
            """Returns (sess_opts, load_path). If a pre-optimized file exists, loads
            that directly with optimization disabled; otherwise runs ORT_ENABLE_ALL
            and serializes the result for next run. TRT providers cannot serialize
            (compiled nodes), so pass serialize_optimized=False for those.
            BOXER_DISABLE_OPTIMIZED_CACHE=1 bypasses cache entirely (Stage 5 off)."""
            opts = ort.SessionOptions()
            if serialize_optimized and not _disable_opt_cache:
                optimized_path = onnx_file_path + ".optimized.onnx"
                if os.path.exists(optimized_path):
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    return opts, optimized_path
                opts.optimized_model_filepath = optimized_path
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return opts, onnx_file_path

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

        # Cache BoxerNet attributes before deleting unused heavy PyTorch modules.
        # These are referenced in forward() and must survive module deletion.
        obj._dino_patch_size = model.dino.patch_size
        obj._with_ray = model.with_ray

        # DinoV3: TensorrtExecutionProvider (fixed shape 1×3×960×960, cache reused across runs)
        # Set trt_fp16_enable=True for ~14% faster inference at ~7% fewer 3D detections.
        # Set trt_fp16_enable=False (or remove the key) for full FP32 accuracy.
        # trt_max_workspace_size: TRT ビルド時のワークスペース上限 (デフォルト ~4 GB → 1 GB に制限)
        # 推論速度には影響せず、ピーク RAM を抑制する
        trt_cache = os.environ.get('ORT_TENSORRT_CACHE_PATH', '/workspace/trt_cache')
        # BOXER_DINO_FP16=1 enables FP16 TRT engine for Dino (-2〜3 GB RAM, ~14% faster,
        # but past measurement showed -7% 3D detections as accuracy cost).
        _dino_fp16 = os.environ.get("BOXER_DINO_FP16") == "1"
        dino_trt_providers = (
            [('TensorrtExecutionProvider', {
                'trt_fp16_enable': _dino_fp16,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': trt_cache,
                'trt_max_workspace_size': _trt_workspace_bytes(),  # BOXER_TRT_WORKSPACE_MB (default 1024)
            }), ('CUDAExecutionProvider', _cuda_ep_opts()), 'CPUExecutionProvider']
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        # BoxerNetCore: CUDAExecutionProvider only.
        # The M dimension (number of 2D detections) is dynamic and changes every frame,
        # which causes TensorrtExecutionProvider to recompile the engine per unique M value (~27 s each).
        # DinoV3 uses TRT which compiles nodes into engines → cannot serialize optimized ONNX.
        dino_opts, dino_load = _make_sess_opts_and_path(dino_path, serialize_optimized=False)
        core_opts, core_load = _make_sess_opts_and_path(core_path)
        obj._dino_session = ort.InferenceSession(dino_load, dino_opts, providers=dino_trt_providers)
        _log_mem("after Dino session init")
        obj._core_session = ort.InferenceSession(core_load, core_opts, providers=cuda_providers)
        _log_mem("after Core session init")

        # IOBinding for Dino (fixed shape, persistent output) and Core (dynamic M, per-frame output).
        # Eliminates .cpu().numpy() round trips that caused ~30 ms/frame of sync-point cost
        # on Jetson (unified memory — cost is sync/GIL, not bandwidth).
        obj._use_io_binding = (device == "cuda")
        if obj._use_io_binding:
            obj._dino_io_binding = obj._dino_session.io_binding()
            obj._core_io_binding = obj._core_session.io_binding()
            # Dino output shape is (1, 384, H/patch, W/patch). For 960×960, patch=16 → (1, 384, 60, 60).
            # Allocated lazily at first _run_dino_onnx call once H, W are known.
            obj._dino_out_buf = None
            obj._dino_out_shape = None

        # Release PyTorch weights for modules replaced by ONNX sessions.
        # Kept in _original_model: head (PyTorch, used post-ONNX), prepare_inputs, process_camera.
        import gc
        for name in ('dino', 'input2emb', 'query2emb', 'self_attn', 'cross_attn'):
            if hasattr(model, name):
                delattr(model, name)
            if name in obj.__dict__:
                del obj.__dict__[name]
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

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
        patch_size = self._dino_patch_size
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

        if self._use_io_binding:
            img = img.contiguous().float()
            dev_id = img.device.index or 0
            _, _, iH, iW = img.shape
            # Dino output shape is fixed by input size. Allocate/reallocate on first call
            # or when input resolution changes (normally never — Aria is always 960×960).
            out_shape = (B, 384, iH // patch_size, iW // patch_size)
            if self._dino_out_buf is None or self._dino_out_shape != out_shape:
                self._dino_out_buf = torch.empty(
                    out_shape, dtype=torch.float32, device=self._device
                )
                self._dino_out_shape = out_shape
                self._dino_io_binding.bind_output(
                    name="features",
                    device_type="cuda",
                    device_id=dev_id,
                    element_type=np.float32,
                    shape=out_shape,
                    buffer_ptr=self._dino_out_buf.data_ptr(),
                )
            self._dino_io_binding.bind_input(
                name="img",
                device_type="cuda",
                device_id=dev_id,
                element_type=np.float32,
                shape=tuple(img.shape),
                buffer_ptr=img.data_ptr(),
            )
            # Sync torch default stream → ORT stream (Dino uses TRT EP) before read,
            # and ORT stream → torch default stream before post-process reads dino_feat.
            self._dino_io_binding.synchronize_inputs()
            self._dino_session.run_with_iobinding(self._dino_io_binding)
            self._dino_io_binding.synchronize_outputs()
            feat = self._dino_out_buf
        else:
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
        if self._use_io_binding:
            x_combined = x_combined.contiguous().float()
            bb2d_norm = bb2d_norm.contiguous().float()
            dev_id = x_combined.device.index or 0
            # M is dynamic per frame — cannot persist output buffer. Allocate fresh each call.
            # Output shape: (B, M, dim). dim is known from the model output; derive it once via
            # a single CPU-path call at init, or (simpler) get_outputs() after first run.
            B, M, _ = bb2d_norm.shape
            if not hasattr(self, "_core_out_dim"):
                # First call: let ORT allocate output, record dim for future torch.empty sizing.
                x_np = x_combined.detach().cpu().numpy()
                b_np = bb2d_norm.detach().cpu().numpy()
                outputs = self._core_session.run(
                    None, {"x_combined": x_np, "bb2d_norm": b_np}
                )
                self._core_out_dim = outputs[0].shape[-1]
                return torch.from_numpy(outputs[0]).to(self._device)
            out_shape = (B, M, self._core_out_dim)
            out_buf = torch.empty(out_shape, dtype=torch.float32, device=self._device)
            self._core_io_binding.bind_input(
                name="x_combined",
                device_type="cuda",
                device_id=dev_id,
                element_type=np.float32,
                shape=tuple(x_combined.shape),
                buffer_ptr=x_combined.data_ptr(),
            )
            self._core_io_binding.bind_input(
                name="bb2d_norm",
                device_type="cuda",
                device_id=dev_id,
                element_type=np.float32,
                shape=tuple(bb2d_norm.shape),
                buffer_ptr=bb2d_norm.data_ptr(),
            )
            self._core_io_binding.bind_output(
                name="query",
                device_type="cuda",
                device_id=dev_id,
                element_type=np.float32,
                shape=out_shape,
                buffer_ptr=out_buf.data_ptr(),
            )
            self._core_io_binding.synchronize_inputs()
            self._core_session.run_with_iobinding(self._core_io_binding)
            self._core_io_binding.synchronize_outputs()
            return out_buf
        else:
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
            sdp_w, cam, T_wr, H, W, self._dino_patch_size
        )
        sdp_input = sdp_median.reshape(B, -1, fH * fW).permute(0, 2, 1)  # [B, fH*fW, 1]

        x_combined = torch.cat([dino_feat_flat, sdp_input], dim=-1)  # [B, fH*fW, 385]

        # Append Plucker ray encoding if with_ray=True
        if self._with_ray:
            T_vc = T_wv.inverse() @ T_wc
            ray_enc = generate_plucker_encoding(
                B, fH, fW, self._dino_patch_size, cam, T_vc
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
