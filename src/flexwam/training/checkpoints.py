from typing import Any
import torch
import torch.nn as nn
from flexwam.models.architectures.state_conditioning import state_config

from flexwam.models.architectures.view_scoring_head import (
    CANONICAL_HEAD_ARCH,
    SpatialQueryHead,
    arches_compatible,
    normalize_head_arch,
)

HEAD_WEIGHT_KEYS = (
    "feature_dim",
    "pool_grid",
    "num_queries",
    "num_heads",
    "spatial_hidden_dim",
    "branch_output_dim",
    "head_hidden_dim",
)


class CheckpointMixin:
    def state_conditioning_signature(self) -> dict[str, Any]:
        """Architecture contract for full training-state resumes."""
        return {
            "mode": self.state_conditioning_mode,
            "video": dict(getattr(self.video_expert, "state_conditioning", state_config(None, video=True))),
            "proprio_dim": self.proprio_dim,
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        payload["state_proj"] = self.state_proj.state_dict()
        if self.context_adapter is not None:
            payload["context_adapter"] = self.context_adapter.state_dict()
        payload["action_camera_router"] = self.action_camera_router.state_dict()
        payload["action_router_arch"] = str(getattr(self, "action_router_arch", CANONICAL_HEAD_ARCH))
        payload["action_router_feature_layer_index"] = int(self.action_router_feature_layer_index)
        payload["action_router_config"] = dict(getattr(self, "action_router_config", {}))
        payload["action_camera_critic"] = self.action_camera_critic.state_dict()
        payload["action_critic_arch"] = str(getattr(self, "action_critic_arch", CANONICAL_HEAD_ARCH))
        payload["action_critic_feature_layer_index"] = int(self.action_critic_feature_layer_index)
        payload["action_critic_config"] = dict(getattr(self, "action_critic_config", {}))
        video_state = getattr(self.video_expert, "state_conditioning", state_config(None, video=True))
        if self.state_conditioning_mode == "shared" or video_state["enabled"]:
            payload["state_conditioning"] = {
                "version": 2, "mode": self.state_conditioning_mode,
                "video": dict(video_state), "proprio_dim": self.proprio_dim,
            }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def _payload_head_config(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
        stored = payload.get(f"{prefix}_config")
        if not isinstance(stored, dict) or not stored:
            raise ValueError(f"Checkpoint is missing `{prefix}_config`.")
        config = dict(stored)
        if "arch" in config:
            config["arch"] = normalize_head_arch(config["arch"])
        pool_grid = config.get("pool_grid")
        if pool_grid is not None:
            config["pool_grid"] = [int(pool_grid[0]), int(pool_grid[1])]
        return config

    def _head_mismatch(self, module: SpatialQueryHead, layer_index: int, payload_config: dict[str, Any]) -> str | None:
        current = module.network_config()
        current["feature_layer_index"] = int(layer_index)
        if "arch" in payload_config and not arches_compatible(payload_config.get("arch"), current["arch"]):
            return f"arch checkpoint={payload_config.get('arch')!r} current={current['arch']!r}"
        payload_layer = payload_config.get("feature_layer_index")
        if payload_layer is not None and int(payload_layer) != int(layer_index):
            return f"feature_layer_index checkpoint={payload_layer} current={layer_index}"
        for key in HEAD_WEIGHT_KEYS:
            if key not in payload_config:
                continue
            left, right = payload_config[key], current[key]
            if key == "pool_grid":
                left = [int(left[0]), int(left[1])]
                right = [int(right[0]), int(right[1])]
            if left != right:
                return f"{key} checkpoint={left!r} current={right!r}"
        return None

    def _load_camera_head_state(
        self,
        module: nn.Module,
        payload: dict[str, Any],
        key: str,
        *,
        prefix: str,
        layer_index: int,
    ) -> None:
        if key not in payload:
            raise ValueError(f"Checkpoint is missing `{key}` weights.")
        payload_config = self._payload_head_config(payload, prefix)
        mismatch = self._head_mismatch(module, layer_index, payload_config)
        if mismatch is None:
            # Preflight before load_state_dict: a failing strict load can already
            # have overwritten matching parameters in an otherwise incompatible head.
            expected = module.state_dict()
            incoming = payload[key]
            missing = sorted(expected.keys() - incoming.keys())
            unexpected = sorted(incoming.keys() - expected.keys())
            if missing or unexpected:
                mismatch = f"state keys missing={missing} unexpected={unexpected}"
            else:
                for name, tensor in expected.items():
                    if incoming[name].shape != tensor.shape:
                        mismatch = (f"{name} shape checkpoint={tuple(incoming[name].shape)} "
                                    f"current={tuple(tensor.shape)}")
                        break
        if mismatch is not None:
            raise ValueError(f"Incompatible `{key}` checkpoint ({mismatch}).")
        module.load_state_dict(payload[key], strict=True)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        adapter_state = payload.get("context_adapter")
        if (adapter_state is not None) != (self.context_adapter is not None):
            raise ValueError("Checkpoint context adapter does not match context_input_dim/text_dim configuration.")
        if self.context_adapter is not None:
            self.context_adapter.load_state_dict(adapter_state, strict=True)
        if "mot" not in payload:
            raise ValueError(f"Checkpoint is missing `mot` weights: {path}")
        saved_state = payload.get("state_conditioning", {})
        if saved_state and saved_state.get("version") != 2:
            raise ValueError("Unsupported state conditioning checkpoint version.")
        saved_mode = saved_state.get("mode", "separate")
        if saved_mode != self.state_conditioning_mode:
            raise ValueError(
                f"State conditioning mode mismatch: checkpoint={saved_mode}, "
                f"model={self.state_conditioning_mode}."
            )
        saved_video_enabled = any(key.startswith("mixtures.video.state_proj.") for key in payload["mot"])
        video_enabled = getattr(self.video_expert, "state_proj", None) is not None
        if saved_mode == "separate" and saved_state and bool(saved_state.get("video", {}).get("enabled", False)) != saved_video_enabled:
            raise ValueError("Video state checkpoint metadata does not match saved projection weights.")
        if saved_mode == "shared" and (
            "state_proj" not in payload
            or saved_state.get("video") != self.video_expert.state_conditioning
            or saved_state.get("proprio_dim") != self.proprio_dim
        ):
            raise ValueError("Shared state checkpoint does not match the model.")
        if saved_video_enabled != video_enabled:
            raise ValueError("Checkpoint video state conditioning does not match the model config.")
        if saved_video_enabled and (
            saved_state.get("video") != self.video_expert.state_conditioning
            or saved_state.get("proprio_dim") != self.proprio_dim
        ):
            raise ValueError("Video state conditioning checkpoint config mismatch.")
        self.mot.load_state_dict(payload["mot"], strict=True)
        if "state_proj" not in payload:
            raise ValueError("Checkpoint is missing `state_proj` weights.")
        self.state_proj.load_state_dict(payload["state_proj"], strict=True)
        self._load_camera_head_state(
            self.action_camera_router,
            payload,
            "action_camera_router",
            prefix="action_router",
            layer_index=int(self.action_router_feature_layer_index),
        )
        self._load_camera_head_state(
            self.action_camera_critic,
            payload,
            "action_camera_critic",
            prefix="action_critic",
            layer_index=int(self.action_critic_feature_layer_index),
        )
        if optimizer is not None:
            if "optimizer" not in payload:
                raise ValueError("Checkpoint is missing optimizer state.")
            optimizer.load_state_dict(payload["optimizer"])
        return payload
