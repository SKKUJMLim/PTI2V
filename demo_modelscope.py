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

from collections import deque

from world_state import load_vjepa2_encoder, extract_world_state_window, compute_drift


print(torch.cuda.is_available())
print("Num GPUs available: ", torch.cuda.device_count())

# PARAMETER SETTINGS
# Choose your GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Text input examples
# input = "A mesmerizing display of the northern lights in the Arctic."
input = "A panda is dancing in the Times Square."

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
# print(img_path)
print(input, postfix)
print("video_cond:", add_vid_cond, "ddpm_inv:", use_ddpm_inversion, "#resample:", resample_iter)

# default parameters
IMG_H = 256
IMG_W = 256
NUM_FRAMES = 16
NUM_COND_FRAMES = 15

t2v_pipeline = TextToVideoSynthesisPipeline(**config)
processed_input = t2v_pipeline.preprocess([input])


for sample_idx in range(NUM_SAMPLES):

    newpostfix = postfix + "-%02d" % sample_idx
    # vid_tensor = t2v_pipeline.preprocess_vid(deepcopy(cond_vid_npy))
    # new_output_tensor = vid_tensor.clone().detach().cpu() # 최종 비디오

    output_filename = input.replace(" ", "_")[:-1] + "%s-%02d.gif" % (newpostfix, NUM_NEW_FRAMES)
    # output_filename = input.replace(" ", "_")[:-1] + "%s-%02d.mp4" % (newpostfix, NUM_NEW_FRAMES)
    video_name = os.path.basename(output_filename)[:-4]

    save_img_dir = os.path.join(output_img_dir, video_name)
    os.makedirs(save_img_dir, exist_ok=True)

    output = t2v_pipeline.forward(
        processed_input,
        # vid=vid_tensor,
        # add_vid_cond=add_vid_cond,
        # use_ddpm_inversion=False,
        # resample_iter=resample_iter,
        ddim_step=ddim_step,
        guide_scale=9.0,
    )

    output_video = t2v_pipeline.postprocess(
        output, os.path.join(output_dir, output_filename)
    )

    print("saving to", save_img_dir)
    print("saving video to", os.path.join(output_dir, output_filename))
