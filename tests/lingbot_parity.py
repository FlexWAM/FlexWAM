"""Small LingBot stand-in for the unit tests.

Uses the real transformer, router, and critic with narrow widths and a deterministic
VAE stand-in. No downloaded weights or robot hardware are needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from flexwam.models.architectures.flexwam import FlexWAM
from flexwam.models.architectures.mot import MoT
from flexwam.models.backbone.lingbot_video.lingbot_video import LingBotVideoTransformer3DModel
from flexwam.models.backbone.lingbot_video.action_dit import LingBotActionDiT


class FixtureVAE(torch.nn.Module):
    upsampling_factor = 8
    temporal_downsample_factor = 4
    model = SimpleNamespace(z_dim=2)

    def encode(self, video, device=None, **kwargs):
        if not torch.is_tensor(video):
            video = torch.stack(list(video))
        video = video.to(device=device)[:, :2, ::4]
        b, c, t, h, w = video.shape
        return F.avg_pool2d(video.transpose(1, 2).reshape(b*t, c, h, w), 8).reshape(b, t, c, h//8, w//8).transpose(1, 2)

    def decode(self, latents, device=None, **kwargs):
        latents = latents.to(device=device)
        b, c, t, h, w = latents.shape
        pixels = F.interpolate(latents.transpose(1, 2).reshape(b*t, c, h, w).float(), scale_factor=8, mode='nearest')
        pixels = torch.cat([pixels, pixels[:, :1]], dim=1)
        return pixels.reshape(b, t, 3, h*8, w*8).transpose(1, 2).to(latents.dtype)


def make_model(device, dtype, critic_config=None, router_cf_config=None, *,
               video_state_config=None, state_conditioning=None):
    torch.manual_seed(117)
    video = LingBotVideoTransformer3DModel(
        hidden_size=48, intermediate_size=80, depth=20,
        num_attention_heads=2, in_channels=2, out_channels=2,
        text_dim=16, freq_dim=16, axes_dims=(8,8,8), axes_lens=(64,32,32),
        num_views=3, use_camera_embedding=True, cross_view_attention_interval=4,
    )
    video.video_attention_mask_mode = 'first_frame_causal'
    video.use_gradient_checkpointing = False
    action = LingBotActionDiT(action_dim=3, hidden_size=48, intermediate_size=80,
        depth=20, num_attention_heads=2, freq_dim=16, axes_dims=(8,8,8), axes_lens=(64,32,32))
    model = FlexWAM(video_expert=video, action_expert=action,
        mot=MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=False),
        vae=FixtureVAE(), text_dim=16, proprio_dim=3, device=device,
        torch_dtype=dtype, action_train_shift=5.0, action_infer_shift=5.0,
        action_router_config={'num_views':3, 'feature_layer_index':18,
        'feature_dim':16, 'pool_grid':[2,2], 'num_queries':2, 'num_heads':2,
        'spatial_hidden_dim':24, 'branch_output_dim':16, 'head_hidden_dim':24},
        action_critic_config=critic_config,
        video_state_config=video_state_config,
        state_conditioning=state_conditioning,
        router_cf_config={'interval':1, 'uniform_probe':True,
        'action_sigma_min':0.6, 'action_sigma_max':1.0, **(router_cf_config or {})})
    model = model.to(device=device, dtype=dtype)
    # Exercise non-uniform routing instead of a zero-initialized head.
    with torch.no_grad():
        model.action_camera_router.head[-1].weight.normal_(std=0.05)
        model.action_camera_critic.head[-1].weight.normal_(std=0.05)
    return model
