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
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# NIQE model address
niqe_model_path = "./results/pretrained_models/niqe_model.mat"
lpips_net = 'alex'
# Model architecture name
d_arch_name = "discriminator"
g_arch_name = "rrdbnet_x4"
# Model arch config
in_channels = 3
out_channels = 3
channels = 64
growth_channels = 32
num_blocks = 23
upscale_factor = 4
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "train_ESRGAN_x4_bubbles"

# MLflow
experience_name = 'ESRGAN_x4' # each name is associated with unique id
run_name = 'train_30batches'
run_id = '2f4d7eef4e784292892a39599f91f03c' # used to resume runs
tags = ''
description = 'ESRGAN base model trained on 10 epochs on the Bubble dataset'

if mode == "train":
    print("Train")
    # Dataset address
    '''
    train_gt_images_dir = f"./data/DIV2K/ESRGAN/train"

    valid_gt_images_dir = f"./data/DIV2K/ESRGAN/validHR"
    valid_lr_images_dir = f"./data/DIV2K/ESRGAN/validLR"

    test_gt_images_dir = f"./data/BSDS100/GTmod12"
    test_lr_images_dir = f"./data/BSDS100/LRbicx{upscale_factor}"
    '''

    train_gt_images_dir = f"./data/Bubbles/train"

    valid_gt_images_dir = f"./data/Bubbles/valid"


    gt_image_size = 128
    batch_size = 16
    num_workers = 2

    # The address to load the pretrained model
    pretrained_d_model_weights_path = "./results/Discriminator/Discrminator_x4-DFO2K-e74d7ca1.pth.tar"
    pretrained_g_model_weights_path = "./results/RRDBNet_x4/RRDBNet_x4-DFO2K-2e2a91f4.pth.tar"
    #pretrained_d_model_weights_path = "./mlruns/504830559517801873/ba82541065944037bec4ce660262710f/artifacts/d_model_best"
    #pretrained_g_model_weights_path = "./mlruns/504830559517801873/ba82541065944037bec4ce660262710f/artifacts/g_model_best"


    # Incremental training and migration training
    #resume_d_model_weights_path = f"./results/train_ESRGAN_x4/d_best.pth.tar"
    resume_d_model_weights_path = f""
    #resume_g_model_weights_path = f"./results/train_ESRGAN_x4/g_best.pth.tar"
    resume_g_model_weights_path = f""

    # Total num epochs (400,000 iters)
    epochs = 10
    print("Total Epochs -> "+str(epochs))

    # Loss function weight
    pixel_weight = 0.01
    content_weight = 1.0
    adversarial_weight = 0.005

    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.34"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 100

if mode == "test":
    print("Test")
    # Test data address
    #lr_dir = f"./data/BSDS100/LRbicx{upscale_factor}"
    #sr_dir = f"./results/test/{exp_name}"
    
    #gt_image_size = 480 # bubbles
    #lr_dir = f"./data/Bubbles/bubblesLR"
    #sr_dir = f"./data/Bubbles/testSR"
    gt_dir = f"./data/Bubbles/test"

    #g_model_weights_path = "./results/pretrained_models/ESRGAN_x4-DFO2K-25393df7.pth.tar"
    g_model_weights_path = "./mlruns/458631362597146827/2f4d7eef4e784292892a39599f91f03c/artifacts/g_model"
