# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
print("Device: "+str(device))
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# NIQE model address
niqe_model_path = "./results/pretrained_models/niqe_model.mat"
lpips_net = 'alex'
# model schema name
d_model_arch_name = "discriminator_unet"
g_model_arch_name = "rrdbnet_x4"
# Discriminator model parameters
d_in_channels = 3
d_out_channels = 1
d_channels = 64
# Generator model parameters
g_in_channels = 3
g_out_channels = 3
g_channels = 64
g_growth_channels = 32
g_num_rrdb = 23

degradation_model_parameters_dict = {
    "sinc_kernel_size": 21,
    "gaussian_kernel_range": [7, 9, 11, 13, 15, 17, 19, 21],
    "gaussian_kernel_type": ["isotropic", "anisotropic",
                             "generalized_isotropic", "generalized_anisotropic",
                             "plateau_isotropic", "plateau_anisotropic"],
    # First-order degradation parameters
    "gaussian_kernel_probability1": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_kernel_probability1": 0.1,
    "gaussian_sigma_range1": [0.2, 3],
    "generalized_kernel_beta_range1": [0.5, 4],
    "plateau_kernel_beta_range1": [1, 2],
    # Second-order degradation parameters
    "gaussian_kernel_probability2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_kernel_probability2": 0.1,
    "gaussian_sigma_range2": [0.2, 1.5],
    "generalized_kernel_beta_range2": [0.5, 4],
    "plateau_kernel_beta_range2": [1, 2],
    "sinc_kernel_probability3": 0.8,
}

degradation_process_parameters_dict = {
    # The probability of triggering a first-order degenerate operation
    "first_blur_probability": 1.0,
    # First-Order Degenerate Operating Parameters
    "resize_probability1": [0.2, 0.7, 0.1],
    "resize_range1": [0.15, 1.5],
    "gray_noise_probability1": 0.4,
    "gaussian_noise_probability1": 0.5,
    "noise_range1": [1, 30],
    "poisson_scale_range1": [0.05, 3],
    "jpeg_range1": [30, 95],
    # The probability of triggering a second-order degenerate operation
    "second_blur_probability": 0.8,
    # Second-Order Degenerate Operating Parameters
    "resize_probability2": [0.3, 0.4, 0.3],
    "resize_range2": [0.3, 1.2],
    "gray_noise_probability2": 0.4,
    "gaussian_noise_probability2": 0.5,
    "noise_range2": [1, 25],
    "poisson_scale_range2": [0.05, 2.5],
    "jpeg_range2": [30, 95],
}

# model magnification
upscale_factor = 4
# Current configuration parameter method
mode = "train"
optimizing_metric = "LPIPS"
loadsFromMlrun = True
# Experiment name, easy to save weights and log files
exp_name = "RealESRGAN_x4-DIV2K"

# MLflow
experience_name = 'RealESRGAN_x4_bubbles' # each name is associated with unique id
run_name = 'realesrgan_bubbles_10epochs_lpips'
run_id = '' # used to resume runs
tags = ''
description = 'RealEsrgan upscale 4 default degradation function. Focus on LPIPS, 10 epochs.'


if mode == "train":
    print("Train")
    
    # Dataset address
    train_gt_images_dir = f"../data/Bubbles/train"

    test_gt_images_dir = f"../data/Bubbles/valid"


    gt_image_size = 256
    batch_size = 48
    num_workers = 4

    if loadsFromMlrun:
        print("Loading Mlrun models...")
        #pretrained_d_model_weights_path = "./mlruns/815542563266978794/803fa4c4bdad488bbd2a72649585de2c/artifacts/best_d_model"
        #pretrained_g_model_weights_path = "./mlruns/815542563266978794/803fa4c4bdad488bbd2a72649585de2c/artifacts/best_g_model"
    else:
        print("Loading basic pretrained models...")
        pretrained_d_model_weights_path = "./results/pretrained_models/Real-ESRGAN/Discriminator_x4-DFO2K-460b1766.pth.tar"

        pretrained_g_model_weights_path = "./results/pretrained_models/Real-ESRGAN/RealESRGAN_x4-DFO2K-678bf481.pth.tar"

    # Define this parameter when training is interrupted or migrated
    resume_d_model_weights_path = f""
    resume_g_model_weights_path = f""

    # Total num epochs
    epochs = 15

    # Loss function weights
    pixel_weight = [1.0]
    content_weight = [0.1, 0.1, 1.0, 1.0, 1.0]
    adversarial_weight = [0.1]

    # Feature loss model parameters
    feature_model_extractor_nodes = ["features.2", "features.7", "features.16", "features.25", "features.34"]
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # optimizer parameters
    model_lr = 1e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-4  # Ensure mixed precision training does not appear Nan
    model_weight_decay = 0.0

    # EMA moving average model parameters
    model_ema_decay = 0.999

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 50
    test_print_frequency = 200

if mode == "test":
    print("Test")

    save_images = True

    gt_dir = f"../data/Bubbles/test"

    g_model_weights_path = "./results/pretrained_models/RealESRGAN_x4-DFO2K-678bf481.pth.tar"
