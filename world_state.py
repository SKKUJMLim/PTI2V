# world_state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@torch.no_grad()
def load_vjepa2_encoder(
    device: Union[str, torch.device],
    repo: str = "facebookresearch/vjepa2",
    model_name: str = "vjepa2_vit_giant",
) -> torch.nn.Module:
    """
    Load V-JEPA2 encoder via torch.hub and move to device.
    Returns encoder in eval() mode.
    """

    loaded = torch.hub.load(repo, model_name)

    if isinstance(loaded, tuple):
        encoder = loaded[0]  # encoder만 사용
    else:
        encoder = loaded

    encoder = encoder.to(device).eval()

    return encoder

    # # ---------------- V-JEPA2 (HF) load ----------------
    # processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")
    # loaded = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant")
    #
    # if isinstance(loaded, tuple):
    #     vjepa2_encoder = loaded[0]  # encoder만 사용
    # else:
    #     vjepa2_encoder = loaded
    #
    # vjepa2_encoder = vjepa2_encoder.to(t2v_pipeline.model.device).eval()
    # # ---------------------------------------------------



@torch.no_grad()
def extract_world_state_pair(
    prev_bchw: torch.Tensor,
    curr_bchw: torch.Tensor,
    encoder: torch.nn.Module,
    *,
    assume_0_255: bool = True,
    return_cpu: bool = True,
) -> torch.Tensor:
    """
    prev_bchw, curr_bchw: (B,3,H,W). Usually float32 in [0,255] from TI2V autoencoder decode.
    encoder expects (B,3,T,H,W) with T>=2, so we stack 2 frames.

    Returns: (B,D) pooled world state (by default on CPU).
    """
    assert prev_bchw.shape == curr_bchw.shape
    assert prev_bchw.dim() == 4 and prev_bchw.size(1) == 3, f"got {tuple(prev_bchw.shape)}"

    if assume_0_255:
        prev = prev_bchw.float() / 255.0
        curr = curr_bchw.float() / 255.0
    else:
        prev = prev_bchw.float()
        curr = curr_bchw.float()

    x = torch.stack([prev, curr], dim=2)  # (B,3,2,H,W)
    x = x.to(next(encoder.parameters()).device)

    out = encoder(x)
    if isinstance(out, (tuple, list)):
        out = out[0]

    w = _pool_encoder_output_to_bd(out)

    if return_cpu:
        w = w.detach().cpu()
    return w


@torch.no_grad()
def extract_world_state_window(
    frames_bchw: List[torch.Tensor],
    encoder: torch.nn.Module,
    *,
    assume_0_255: bool = True,
    return_cpu: bool = True,
    use_last_timestep: bool = True,
) -> torch.Tensor:
    """
    frames_bchw: list of (B,3,H,W), length T>=2
    Returns: (B,D) world state. By default, uses tokens of the LAST timestep if encoder returns (B,T,N,D).
    """
    assert len(frames_bchw) >= 2, "V-JEPA2 requires T >= 2"
    for f in frames_bchw:
        assert f.dim() == 4 and f.size(1) == 3, f"bad frame shape {tuple(f.shape)}"

    if assume_0_255:
        x = torch.stack([f.float() / 255.0 for f in frames_bchw], dim=2)  # (B,3,T,H,W)
    else:
        x = torch.stack([f.float() for f in frames_bchw], dim=2)

    x = x.to(next(encoder.parameters()).device)

    out = encoder(x)
    if isinstance(out, (tuple, list)):
        out = out[0]

    # If (B,T,N,D) and use_last_timestep: pool only last timestep tokens -> more "current-frame" like
    if out.dim() == 4 and use_last_timestep:
        w = out[:, -1].mean(dim=1)  # (B,D)
    else:
        w = _pool_encoder_output_to_bd(out)

    if return_cpu:
        w = w.detach().cpu()

    return w


def compute_drift(
    w_prev: torch.Tensor,
    w_curr: torch.Tensor,
) -> Tuple[float, float]:
    """
    w_prev, w_curr: (B,D) on CPU or GPU (same device)
    Returns: (l2_mean, cos_drift_mean) where cos_drift = 1 - cosine_similarity
    """
    l2 = torch.norm(w_curr - w_prev, dim=-1).mean().item()
    cos = F.cosine_similarity(w_curr, w_prev, dim=-1).mean().item()
    return l2, (1.0 - cos)


def _pool_encoder_output_to_bd(out: torch.Tensor) -> torch.Tensor:
    """
    Pool encoder output to (B,D) robustly.
    Supports:
      - (B,T,N,D): mean over N then mean over T
      - (B,N,D): mean over N
      - (B,D): passthrough
    """
    if out.dim() == 4:          # (B,T,N,D)
        return out.mean(dim=2).mean(dim=1)
    if out.dim() == 3:          # (B,N,D)
        return out.mean(dim=1)
    if out.dim() == 2:          # (B,D)
        return out
    raise RuntimeError(f"Unexpected encoder output shape: {tuple(out.shape)}")
