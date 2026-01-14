# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2023 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-2-Clause
#
# Code adapted from https://github.com/nihaomiao/CVPR23_LFDM/tree/main/demo -- BSD-2-Clause License

# Demo for TI2V-Zero

import os
from copy import deepcopy

import imageio
import numpy as np
import torch
from PIL import Image

from modelscope_t2v_pipeline import TextToVideoSynthesisPipeline, tensor2vid
from util import center_crop, save_world_state_logs

import torch.nn.functional as F
from collections import deque


print(torch.cuda.is_available())
print("Num GPUs available: ", torch.cuda.device_count())

# PARAMETER SETTINGS
# Choose your GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Text input examples
# input = "A mesmerizing display of the northern lights in the Arctic."
input = "A panda is dancing in the Times Square."

# Image input examples
# img_path = "examples/northern_lights_sd.jpg"
img_path = "./examples/panda_dancing_sd.png"

# After running initialization.py, set the config path to your ModelScope path
config = {"model": "./weights", "device": "gpu"}

# Set your output path
output_dir = "./example-video"
output_img_dir = "./example-image"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

# Set parameters for temporal resampling and DDIM
resample_iter = 4
ddim_step = 10

# Set the number of new frames
NUM_NEW_FRAMES = 15
print("#new_frame:", NUM_NEW_FRAMES)

# Set the number of generated videos
NUM_SAMPLES = 1

postfix = "-resample%02d-s%02d-mean%04d" % (resample_iter, ddim_step, np.random.randint(low=0, high=10000))
add_vid_cond = True
use_ddpm_inversion = True
print(img_path)
print(input, postfix)
print("video_cond:", add_vid_cond, "ddpm_inv:", use_ddpm_inversion, "#resample:", resample_iter)

# default parameters
IMG_H = 256
IMG_W = 256
NUM_FRAMES = 16
NUM_COND_FRAMES = 15

# read image
first_img_npy = imageio.v2.imread(img_path)
# crop image
first_img_npy = center_crop(first_img_npy)
# resize image
first_img_npy = np.asarray(Image.fromarray(first_img_npy).resize((IMG_H, IMG_W)))
# repeat image
first_img_npy_list = [first_img_npy for i in range(NUM_COND_FRAMES)]
cond_vid_npy = np.stack(first_img_npy_list, axis=0)
t2v_pipeline = TextToVideoSynthesisPipeline(**config)
processed_input = t2v_pipeline.preprocess([input])

# ---------------- V-JEPA2 (HF) load ----------------
processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")
loaded    = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant")

if isinstance(loaded, tuple):
    vjepa2_encoder = loaded[0]    # encoder만 사용
else:
    vjepa2_encoder = loaded

vjepa2_encoder = vjepa2_encoder.to(t2v_pipeline.model.device).eval()
# ---------------------------------------------------


@torch.no_grad()
def extract_world_state_pair(prev_bchw: torch.Tensor, curr_bchw: torch.Tensor) -> torch.Tensor:
    """
    prev_bchw, curr_bchw: (B,3,H,W), value range 0..255 (float32 ok)
    return: (B,D) world-state vector for 'current' frame (approx)
    """
    assert prev_bchw.shape == curr_bchw.shape
    assert prev_bchw.dim() == 4 and prev_bchw.size(1) == 3

    # 0..255 -> 0..1
    prev = prev_bchw.float() / 255.0
    curr = curr_bchw.float() / 255.0

    # (B,3,T=2,H,W)
    x = torch.stack([prev, curr], dim=2)

    enc_device = next(vjepa2_encoder.parameters()).device
    x = x.to(enc_device)

    out = vjepa2_encoder(x)
    if isinstance(out, (tuple, list)):
        out = out[0]

    # out shape handling
    # 기대 케이스:
    #  (B, T, N, D) or (B, N, D) or (B, D)
    if out.dim() == 4:
        # (B,T,N,D): 'curr' 프레임(t=last)의 토큰만 풀링
        # out[:, -1] -> (B,N,D)
        w = out[:, -1].mean(dim=1)     # (B,D)
    elif out.dim() == 3:
        # (B,N,D): T축이 이미 합쳐져 나온 경우 -> 그냥 평균
        w = out.mean(dim=1)            # (B,D)
    elif out.dim() == 2:
        w = out
    else:
        raise RuntimeError(f"Unexpected encoder output shape: {tuple(out.shape)}")

    return w.detach().cpu()


@torch.no_grad()
def extract_world_state_window(frames_bchw, vjepa2_encoder) -> torch.Tensor:
    """
    frames_bchw: list[Tensor], each (B,3,H,W), value range 0..255 (float ok)
    returns: (B,D) on CPU
    """
    assert len(frames_bchw) >= 2, "V-JEPA2 requires T >= 2"
    B = frames_bchw[0].size(0)

    # 0..255 -> 0..1 then stack to (B,3,T,H,W)
    x = torch.stack([f.float() / 255.0 for f in frames_bchw], dim=2)

    enc_device = next(vjepa2_encoder.parameters()).device
    x = x.to(enc_device)

    out = vjepa2_encoder(x)
    if isinstance(out, (tuple, list)):
        out = out[0]

    # pool to (B,D)
    if out.dim() == 4:
        # (B,T,N,D) -> last time step tokens -> (B,N,D) -> mean over N
        w = out[:, -1].mean(dim=1)
    elif out.dim() == 3:
        # (B,N,D)
        w = out.mean(dim=1)
    elif out.dim() == 2:
        # (B,D)
        w = out
    else:
        raise RuntimeError(f"Unexpected encoder output shape: {tuple(out.shape)}")

    return w.detach().cpu()


for sample_idx in range(NUM_SAMPLES):

    newpostfix = postfix + "-%02d" % sample_idx
    vid_tensor = t2v_pipeline.preprocess_vid(deepcopy(cond_vid_npy))

    new_output_tensor = vid_tensor.clone().detach().cpu() # 최종 비디오

    output_filename = input.replace(" ", "_")[:-1] + "%s-%02d.gif" % (newpostfix, NUM_NEW_FRAMES)
    video_name = os.path.basename(output_filename)[:-4]
    save_img_dir = os.path.join(output_img_dir, video_name)
    os.makedirs(save_img_dir, exist_ok=True)
    img_name = video_name + "%03d.jpg" % 0
    img_path = os.path.join(save_img_dir, img_name)
    imageio.v2.imsave(img_path, first_img_npy)

    # ---------- [Pair] world state logging: sample-level init ----------
    # SAVE_WORLD_STATE = True
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # frame0 tensor (0..255 float32 GPU)
    # first_frame = torch.from_numpy(first_img_npy.copy()).float().permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    #
    # states = []  # list of (B,D) on CPU
    # drift_l2 = []
    # drift_cos = []
    #
    # prev_frame_for_w = first_frame  # for pair input
    # # 초기 w0는 (frame0, frame0)으로 패딩해서 정의 (T=2 필요)
    # w0 = extract_world_state_pair(first_frame, first_frame)  # (B,D) CPU
    # states.append(w0)
    # print("w0.shape =", w0.shape)
    # -----------------------------------------------------------

    # ---------- [Sliding Window] world state logging: sample-level init ----------
    K = 8  # 추천: 4~8부터 시작
    frame_buf = deque(maxlen=K)

    # first_frame: (1,3,H,W) 0..255 float, GPU
    first_frame = torch.from_numpy(first_img_npy.copy()).float().permute(2, 0, 1).unsqueeze(0).to(t2v_pipeline.model.device)

    # 버퍼를 최소 T=2로 채우기 (초기엔 동일 프레임 복제)
    frame_buf.append(first_frame)
    frame_buf.append(first_frame)
    # -----------------------------------------------------------

    states = []
    drift_l2, drift_cos = [], []

    # 초기 state (t=0에 해당): window 기반
    w0 = extract_world_state_window(list(frame_buf), vjepa2_encoder)
    states.append(w0)
    print("w0.shape =", w0.shape)


    # image-to-video generation
    for i in range(NUM_NEW_FRAMES):
        print("i:", i, input, newpostfix)

        output = t2v_pipeline.forward_with_vid_resample(
            processed_input,
            vid=vid_tensor,
            add_vid_cond=add_vid_cond,
            use_ddpm_inversion=use_ddpm_inversion,
            resample_iter=resample_iter,
            ddim_step=ddim_step,
            guide_scale=9.0,
        )

        with torch.no_grad():
            new_frame = t2v_pipeline.model.autoencoder.decode(output[:, :, -1].cuda())

            # ---------- [Pair] world state logging: frame-level append ----------
            # w = extract_world_state_pair(prev_frame_for_w, new_frame)  # (B,D) CPU
            # states.append(w)
            #
            # w_prev = states[-2]
            # w_curr = states[-1]
            #
            # l2 = torch.norm(w_curr - w_prev, dim=-1).mean().item()
            # cos = F.cosine_similarity(w_curr, w_prev, dim=-1).mean().item()
            # drift_l2.append(l2)
            # drift_cos.append(1.0 - cos)
            #
            # print(f"[state] frame={i + 1:02d}  l2={l2:.6f}  cos_drift={1.0 - cos:.6f}")
            #
            # prev_frame_for_w = new_frame
            # ----------------------------------------------------------

            # ---------- [Sliding Window] world state logging: frame-level append ----------

            # new_frame: (B,3,H,W) 0..255 float on GPU
            frame_buf.append(new_frame)

            # 버퍼 길이가 2 미만일 일은 없지만, 안전장치
            if len(frame_buf) >= 2:
                w = extract_world_state_window(list(frame_buf), vjepa2_encoder)
                states.append(w)

                w_prev = states[-2]
                w_curr = states[-1]

                l2 = torch.norm(w_curr - w_prev, dim=-1).mean().item()
                cos = F.cosine_similarity(w_curr, w_prev, dim=-1).mean().item()

                drift_l2.append(l2)
                drift_cos.append(1.0 - cos)

                print(f"[state@K={K}] frame={i + 1:02d}  l2={l2:.6f}  cos_drift={1.0 - cos:.6f}")
            # ----------------------------------------------------------

        # ---------- [Pair] world state logging: frame-level append ----------
        # SAVE_WORLD_STATE = True  # False로 두면 저장 안 함
        # if SAVE_WORLD_STATE:
        #     save_world_state_logs(
        #         save_img_dir,
        #         states,
        #         drift_l2,
        #         drift_cos,
        #     )
        # ----------------------------------------------------------

        new_frame = new_frame.data.cpu().unsqueeze(dim=2)
        img_npy = tensor2vid(new_frame.clone().detach())[0]
        img_name = video_name + "%03d.jpg" % (i + 1)
        img_path = os.path.join(save_img_dir, img_name)
        imageio.v2.imsave(img_path, img_npy)
        new_output_tensor = torch.cat((new_output_tensor, new_frame), dim=2)
        vid_tensor = new_output_tensor[:, :, (i + 1) :]
        assert vid_tensor.size(2) == NUM_COND_FRAMES

    SAVE_WORLD_STATE = True
    if SAVE_WORLD_STATE:
        save_world_state_logs(save_img_dir, states, drift_l2, drift_cos)

    output_video = t2v_pipeline.postprocess(
        new_output_tensor[:, :, (NUM_COND_FRAMES - 1) :], os.path.join(output_dir, output_filename)
    )

    print("saving to", save_img_dir)
    print("saving video to", os.path.join(output_dir, output_filename))
