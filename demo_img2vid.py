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
from util import center_crop

from collections import deque
from energy.jepa_score import load_vjepa2_encoder

import torch
from energy.jepa_score import jepa_energy_fd, fd_hutchinson_trace_jtj



print(torch.cuda.is_available())
print("Num GPUs available: ", torch.cuda.device_count())

# PARAMETER SETTINGS
# Choose your GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Text input examples
# input = "A mesmerizing display of the northern lights in the Arctic."
# input = "A panda is dancing in the Times Square."
# input = "A basketball free falls in the air"
# input = "A 30lb kettlebell and a green piece of paper are lowered onto two pillows."
input = "A lit match is being lowered into a glass of water."
# input = "A basketball falls from the air to the floor."

# Image input examples
# img_path = "examples/northern_lights_sd.jpg"
# img_path = "./examples/panda_dancing_sd.png"
# img_path= "./examples/ball.png"
# img_path= "./examples/orange.jpg"
img_path= "./examples/cup.jpg"

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

##########################################################################################################################################
def make_fd_energy_condition_fn(
    t2v_pipeline,
    vjepa2_encoder,
    method: str = "fd",          # "fd" or "hutch"
    n_dir: int = 2,              # jepa_energy_fd의 n_dir
    n_samples: int = 4,          # fd_hutchinson_trace_jtj의 n_samples
    eps_fd: float = 1e-3,        # encoder-input perturb (jepa_score.py 기본값과 맞추는 쪽 권장)
    eps_xt: float = 1e-3,        # latent xt perturb (SPSA step)
    n_probe_xt: int = 2,         # SPSA probe 개수(xt 방향)
    guide_w: float = 0.05,       # condition_fn 출력 스케일
    pool: str = "mean",
    noise: str = "rademacher",
):
    """
    condition_fn(xt,t,**kwargs) -> (B,4,T,32,32) pseudo-grad
    - Uses FD/Hutch energy estimator (no_grad) and SPSA to approximate ∇_xt E(xt)
    """
    device = t2v_pipeline.model.device
    ae = t2v_pipeline.model.autoencoder
    scale_factor = 0.18215  # modelscope_t2v.py에서 사용하는 값 :contentReference[oaicite:6]{index=6}

    def energy_from_xt(xt: torch.Tensor) -> torch.Tensor:
        device = t2v_pipeline.model.device
        ae = t2v_pipeline.model.autoencoder
        scale_factor = 0.18215

        xt = xt.to(device)
        B, _, T, _, _ = xt.shape

        # (B,4,T,32,32) -> (B*T,4,32,32)
        z = xt.permute(0, 2, 1, 3, 4).contiguous().view(B * T, 4, 32, 32)

        z_unscaled = (1.0 / scale_factor) * z
        rgb = ae.decode(z_unscaled)  # (B*T,3,256,256)

        # (B*T,3,256,256) -> (B,3,T,256,256)
        rgb = rgb.view(B, T, 3, 256, 256).permute(0, 2, 1, 3, 4).contiguous()

        if method == "hutch":
            e = fd_hutchinson_trace_jtj(
                vjepa2_encoder, rgb,
                n_samples=n_samples, noise=noise, pool=pool,
                normalize_r=False, eps_fd=eps_fd
            )
        else:
            e = jepa_energy_fd(vjepa2_encoder, rgb, n_dir=n_dir, eps=eps_fd)

        return e  # (B,)

    def condition_fn(xt, t):
        device = xt.device
        B, C, T, H, W = xt.shape

        with torch.cuda.amp.autocast(enabled=False):  # ### CHANGED: 안정성 위해 fp32로
            xt32 = xt.detach().float()  # ### CHANGED: fp32

            grad_acc = torch.zeros_like(xt32)

            for _ in range(n_probe_xt):
                # u를 전체 xt에 주지 말고, "마지막 프레임"에만 준다
                u = torch.zeros_like(xt32)  # ### CHANGED
                u[:, :, -1] = torch.randn_like(xt32[:, :, -1])  # ### CHANGED

                # (선택) per-sample normalize도 마지막 프레임 기준으로만
                u_last = u[:, :, -1]
                u_last = u_last / (u_last.flatten(1).norm(dim=1, keepdim=True)
                                   .view(B, 1, 1, 1).clamp_min(1e-6))  # ### CHANGED
                u[:, :, -1] = u_last  # ### CHANGED

                e_plus = energy_from_xt(xt32 + eps_xt * u).mean()
                e_minus = energy_from_xt(xt32 - eps_xt * u).mean()

                coef = (e_plus - e_minus) / (2.0 * eps_xt)

                grad_acc = grad_acc + coef * u  # 이제 u가 last frame only라 grad도 last frame only

            grad = grad_acc / float(n_probe_xt)

        # dtype/device 맞춰서 반환
        return (guide_w * grad).to(dtype=xt.dtype, device=device)  # ### CHANGED

    return condition_fn


vjepa2_encoder = load_vjepa2_encoder(device=t2v_pipeline.model.device)
condition_fn = make_fd_energy_condition_fn(
    t2v_pipeline=t2v_pipeline,
    vjepa2_encoder=vjepa2_encoder,
    method="fd",        # 또는 "hutch"
    n_dir=2,            # fd energy
    n_samples=4,        # hutch
    eps_fd=1e-3,
    eps_xt=1e-3,
    n_probe_xt=2,
    guide_w=0.05,
    pool="mean",
    noise="rademacher",
)
##########################################################################################################################################



for sample_idx in range(NUM_SAMPLES):

    newpostfix = postfix + "-%02d" % sample_idx
    vid_tensor = t2v_pipeline.preprocess_vid(deepcopy(cond_vid_npy))
    new_output_tensor = vid_tensor.clone().detach().cpu() # 최종 비디오

    output_filename = input.replace(" ", "_")[:-1] + "%s-%02d.gif" % (newpostfix, NUM_NEW_FRAMES)
    # output_filename = input.replace(" ", "_")[:-1] + "%s-%02d.mp4" % (newpostfix, NUM_NEW_FRAMES)
    video_name = os.path.basename(output_filename)[:-4]

    save_img_dir = os.path.join(output_img_dir, video_name)
    os.makedirs(save_img_dir, exist_ok=True)
    img_name = video_name + "%03d.jpg" % 0
    img_path = os.path.join(save_img_dir, img_name)
    imageio.v2.imsave(img_path, first_img_npy)


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
            condition_fn=condition_fn,
        )

        with torch.no_grad():
            new_frame = t2v_pipeline.model.autoencoder.decode(output[:, :, -1].cuda())


        new_frame = new_frame.data.cpu().unsqueeze(dim=2)
        img_npy = tensor2vid(new_frame.clone().detach())[0]
        img_name = video_name + "%03d.jpg" % (i + 1)
        img_path = os.path.join(save_img_dir, img_name)
        imageio.v2.imsave(img_path, img_npy)
        new_output_tensor = torch.cat((new_output_tensor, new_frame), dim=2)

        vid_tensor = new_output_tensor[:, :, (i + 1) :] #

        assert vid_tensor.size(2) == NUM_COND_FRAMES

    output_video = t2v_pipeline.postprocess(
        new_output_tensor[:, :, (NUM_COND_FRAMES - 1) :], os.path.join(output_dir, output_filename)
    )

    print("saving to", save_img_dir)
    print("saving video to", os.path.join(output_dir, output_filename))
