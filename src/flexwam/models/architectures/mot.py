from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from flexwam.models.backbone.common.attention import flash_attention, modulate, rope_apply
from flexwam.models.architectures.structures import _video_visual_view_slices
from flexwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _video_view_slices(
    video_seq_len: int,
    video_meta: dict[str, Any],
) -> list[tuple[int, int]]:
    num_views = int(video_meta.get("num_views", 1))
    if num_views <= 1:
        raise ValueError(f"`num_views` must be > 1 for per-view routing, got {num_views}.")
    slices = _video_visual_view_slices(video_seq_len, video_meta)
    if str(video_meta.get("prefix_layout", "per_view")) == "global":
        return slices
    # Per-view attention also includes each view's local conditioning prefix.
    prefix_len = int(video_meta.get("prefix_len", 0))
    return [(start - prefix_len, end) for start, end in slices]


_ACTION_VIDEO_QUERY_START_CACHE: dict[tuple[Any, ...], int] = {}


def _is_cuda_stream_capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _action_video_query_start(
    action_attention_mask: torch.Tensor,
    video_seq_len: int,
    action_prefix_len: int,
) -> int:
    """If the state prefix may read video keys, include it in action-to-video queries.

    The result is a Python int used for slicing, so it cannot be computed with
    GPU ``.item()`` while a CUDA graph is being captured. Resolve it once on the
    default stream (or from cache) before capture.
    """
    prefix_len = int(action_prefix_len)
    video_seq_len = int(video_seq_len)
    if prefix_len <= 0:
        return 0
    cache_key = (tuple(action_attention_mask.shape), prefix_len, video_seq_len)
    cached = _ACTION_VIDEO_QUERY_START_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if _is_cuda_stream_capturing():
        raise RuntimeError(
            "Action-to-video query start must be resolved before CUDA graph capture; "
            "reading the attention mask with `.item()` is illegal during capture."
        )
    prefix_to_video = action_attention_mask[:prefix_len, :video_seq_len]
    # Host copy of a tiny static slice; never do this on a capturing stream.
    sees_video = bool(prefix_to_video.detach().to(device="cpu").any().item())
    value = 0 if sees_video else prefix_len
    _ACTION_VIDEO_QUERY_START_CACHE[cache_key] = value
    return value


def resolve_action_video_query_start(
    attention_mask: torch.Tensor,
    video_seq_len: int,
    action_seq_len: int,
    action_prefix_len: int,
) -> int:
    """Prefill `_action_video_query_start` from the joint MoT mask on the host."""
    video_seq_len = int(video_seq_len)
    action_seq_len = int(action_seq_len)
    total_seq_len = video_seq_len + action_seq_len
    if tuple(attention_mask.shape) != (total_seq_len, total_seq_len):
        raise ValueError(
            f"`attention_mask` must have shape [{total_seq_len}, {total_seq_len}], "
            f"got {tuple(attention_mask.shape)}."
        )
    action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
    return _action_video_query_start(action_attention_mask, video_seq_len, action_prefix_len)


def _router_logit_bias_mask(
    attention_mask: torch.Tensor,
    view_slices: list[tuple[int, int]],
    router_scales: torch.Tensor,
    bias_strength: float,
    dtype: torch.dtype,
    router_log_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a float SDPA mask ``[B, 1, Sa, S]`` for one joint action softmax.

    Allowed keys contribute 0. Blocked keys contribute -inf.
    Visible keys of camera ``v`` additionally receive ``bias_strength * log(scale_v)``.
    Action keys and tokens outside the camera slices, including a shared state
    token, stay unbiased. Stable FP32 log-scales take precedence over rounded
    probabilities; -inf marks a removed camera. Legacy scale-only callers use
    non-positive scales to remove cameras, independently of bias_strength.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"`attention_mask` must be [Sa, S], got {tuple(attention_mask.shape)}."
        )
    batch_size = int(router_scales.shape[0])
    query_len, key_len = attention_mask.shape
    scales = router_scales.to(dtype=torch.float32)
    if router_log_scales is not None:
        if router_log_scales.shape != router_scales.shape:
            raise ValueError("Router log-scales must have the same shape as router scales.")
        log_scale = router_log_scales.float()
        available = torch.isfinite(log_scale)
        # Never multiply -inf by zero, including in the backward graph.
        log_scale = torch.where(available, log_scale, torch.zeros_like(log_scale))
    else:
        available = scales > 0
        # Replace absent cameras *before* log: where(..., log(0), ...) still
        # evaluates log(0) and can produce NaN gradients in its inactive branch.
        safe_scale = torch.where(available, scales, torch.ones_like(scales))
        log_scale = safe_scale.clamp_min(torch.finfo(torch.float32).tiny).log()
    bias = float(bias_strength) * log_scale
    neg = float("-inf")
    bias = torch.where(available, bias, torch.full_like(bias, neg))
    bias = bias.to(device=attention_mask.device, dtype=dtype)

    ordered = sorted(enumerate(view_slices), key=lambda item: item[1][0])
    chunks: list[torch.Tensor] = []
    cursor = 0
    for view_idx, (start, end) in ordered:
        if start < cursor or end < start or end > key_len:
            raise ValueError(f"Invalid camera slice {(start, end)} for key length {key_len}.")
        if start > cursor:
            chunks.append(bias.new_zeros(batch_size, start - cursor))
        width = end - start
        chunks.append(bias[:, view_idx].view(batch_size, 1).expand(batch_size, width))
        cursor = end
    if cursor < key_len:
        chunks.append(bias.new_zeros(batch_size, key_len - cursor))
    key_bias = torch.cat(chunks, dim=1).view(batch_size, 1, 1, key_len)

    allowed = attention_mask.to(dtype=torch.bool).view(1, 1, query_len, key_len)
    blocked = key_bias.new_full((), neg)
    return torch.where(allowed, key_bias, blocked)


def _validate_view_router_weights(
    view_router_weights: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
) -> None:
    if view_router_weights.shape != (batch_size, num_views):
        raise ValueError(
            "`view_router_weights` shape mismatch: "
            f"weights={tuple(view_router_weights.shape)}, batch_size={batch_size}, num_views={num_views}."
        )


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        if hasattr(block, "apply_mot_post_block"):
            return block.apply_mot_post_block(
                residual_x=residual_x,
                mixed_attn_out=mixed_attn_out,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context_payload=context_payload,
            )

        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        if hasattr(block, "build_mot_attention_io"):
            return block.build_mot_attention_io(
                expert=expert,
                x=x,
                freqs=freqs,
                t_mod=t_mod,
            )

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def forward_video(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        snapshot_layer_indices: Sequence[int] = (),
        cross_view_attention_mask: Optional[torch.Tensor] = None,
        cross_view_layer_indices: Sequence[int] = (),
    ) -> dict[str, Any]:
        """Run the video expert only: per-layer K/V, final tokens, optional layer snapshots.

        K/V are taken from each layer's *input* (same as `prefill_video_cache`).
        Snapshots are taken from that layer's *output*. Layers listed in
        `cross_view_layer_indices` use `cross_view_attention_mask`; all others
        use the view-local `video_attention_mask`. Layer indices are 0-based
        (`18` is `blocks[18]`).
        """
        video_seq_len = video_tokens.shape[1]
        for name, mask in (
            ("video_attention_mask", video_attention_mask),
            ("cross_view_attention_mask", cross_view_attention_mask),
        ):
            if mask is not None and mask.shape != (video_seq_len, video_seq_len):
                raise ValueError(
                    f"`{name}` must be [{video_seq_len},{video_seq_len}], got {tuple(mask.shape)}."
                )
        snapshot_set = {int(idx) for idx in snapshot_layer_indices}
        cross_view_set = {int(idx) for idx in cross_view_layer_indices}
        for label, indices in (("snapshot_layer_indices", snapshot_set), ("cross_view_layer_indices", cross_view_set)):
            for idx in indices:
                if idx < 0 or idx >= self.num_layers:
                    raise ValueError(f"`{label}` must be in [0, {self.num_layers}), got {idx}.")
        if cross_view_set and cross_view_attention_mask is None:
            raise ValueError("`cross_view_attention_mask` is required when cross-view layers are set.")

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        layer_tokens: dict[int, torch.Tensor] = {}
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=(
                    cross_view_attention_mask if layer_idx in cross_view_set else video_attention_mask
                ),
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            if layer_idx in snapshot_set:
                layer_tokens[layer_idx] = x
            kv_cache.append({"k": k, "v": v})
        return {
            "tokens": x,
            "kv_cache": kv_cache,
            "layer_tokens": layer_tokens,
        }

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Legacy inference path: no cross-view layers, no layer snapshots.
        FlexWAM uses `forward_video` with cross-view layers so K/V match training.
        """
        video_out = self.forward_video(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            snapshot_layer_indices=(),
        )
        return video_out["kv_cache"]

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        video_meta: dict[str, Any],
        action_video_router_weights: torch.Tensor,
        action_attention_stats: Optional[list[dict[str, float]]] = None,
        action_attention_heatmaps: Optional[dict[str, Any]] = None,
        action_prefix_len: int = 0,
        action_attention_mode: str = "logit_bias",
        action_attention_bias_strength: float = 1.0,
        action_video_router_log_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `forward_video`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.
            video_meta: Video layout used to identify camera token ranges.
            action_video_router_weights: Required per-camera mixture scales [B,V].

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if any(
            cache["k"].shape[1] != video_seq_len or cache["v"].shape[1] != video_seq_len
            for cache in video_kv_cache
        ):
            raise ValueError(f"`video_kv_cache` seq len mismatch, expected {video_seq_len}.")
        action_seq_len = int(action_tokens.shape[1])
        action_prefix_len = int(action_prefix_len)
        if action_prefix_len < 0 or action_prefix_len > action_seq_len:
            raise ValueError(
                f"`action_prefix_len` must be in [0,{action_seq_len}], got {action_prefix_len}."
            )
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape != (total_seq_len, total_seq_len):
            raise ValueError(
                f"`attention_mask` must be [{total_seq_len},{total_seq_len}], "
                f"got {tuple(attention_mask.shape)}."
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        if video_meta is None or action_video_router_weights is None:
            raise ValueError("Cached action attention requires `video_meta` and camera router weights.")
        view_slices = _video_view_slices(video_seq_len, video_meta)
        _validate_view_router_weights(
            action_video_router_weights,
            batch_size=action_tokens.shape[0],
            num_views=len(view_slices),
        )
        if action_attention_mode != "logit_bias":
            raise ValueError(
                f"`action_attention_mode` must be 'logit_bias', got {action_attention_mode!r}."
            )
        del action_attention_stats, action_attention_heatmaps
        logit_bias_mask = _router_logit_bias_mask(
            action_attention_mask,
            view_slices,
            action_video_router_weights,
            float(action_attention_bias_strength),
            action_tokens.dtype,
            router_log_scales=action_video_router_log_scales,
        )

        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=torch.cat([k_video, k_action], dim=1),
                v_cat=torch.cat([v_video, v_action], dim=1),
                attention_mask=logit_bias_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x
