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

degradation_process_parameters_dict = {
    "jpeg_prob": 0.9,
    "scale2_prob": 0.25,
    "shuffle_prob": 0.1,
    "use_sharp": False,
}

degradation_process_plus_parameters_dict = {
    "poisson_prob": 0.1,
    "speckle_prob": 0.1,
    "shuffle_prob": 0.1,
    "use_sharp": True,
}

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
print("Device: "+str(device))
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# NIQE model address
niqe_model_path = "./results/pretrained_models/niqe_model.mat"
lpips_net = 'alex'
# Model architecture name
d_model_arch_name = "discriminator_unet"
g_model_arch_name = "bsrgan_x2"
# DiscriminatorUNet configure
d_in_channels = 3
d_out_channels = 1
d_channels = 64
# RRDBNet configure
g_in_channels = 3
g_out_channels = 3
g_channels = 64
g_growth_channels = 32
g_num_rrdb = 23
# Upscale factor
upscale_factor = 2
# Current configuration parameter method
mode = "test"
optimizing_metric = "LPIPS"
loadsFromMlrun = False
# Experiment name, easy to save weights and log files
exp_name = "BSRGAN_x2-DIV2K_degradations"

# MLflow
experience_name = 'BSRGANsa_x2_bubbles' # each name is associated with unique id
#experience_name = 'BSRGAN_x2_AirfRANs'
run_name = 'bsrgansa_fromML'
run_id = 'fa8a220b890240d7981796c865e9dfdb' # used to resume runs. Run with other datasets: fa8a220b890240d7981796c865e9dfdb
tags = ''
description = 'BSRGAN with SelfAttention on discriminator only by GPT4. Upscale 2 degradation function id=f7f08d67ddd04543bf87d1a36719cef7. The generator and discriminator are from the pretrained models focus on LPIPS. Higher lr for discrminator than generator.'

experiment_id = '815542563266978794' # for testing

if mode == "train":
    print("Train")
    # Dataset address
    '''train_gt_images_dir = f"./data/DIV2K/BSRGAN/train"

    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"'''

    train_generator = True
    if train_generator is False:
        print("\nGenerator is not being trained!\n")

    print(f'Optimizing for: {optimizing_metric}\n')

    train_gt_images_dir = f"../data/Bubbles/train"
    valid_gt_images_dir = f"../data/Bubbles/valid"
    #train_gt_images_dir = f"../data/AirfRANs/vtuNUT/train"
    #valid_gt_images_dir = f"../data/AirfRANs/vtuNUT/valid"

    crop_image_size = 320
    #crop_image_size = 150
    gt_image_size = int(72 * upscale_factor)
    batch_size = 16
    num_workers = 1

    # Load the address of the pretrained model
    #pretrained_d_model_weights_path = ""
    #pretrained_g_model_weights_path = ""

    if loadsFromMlrun:
        print("Loading Mlrun models...")
        pretrained_d_model_weights_path = "./mlruns/589683858730322811/ef2b79c7176d439d815f3e95c0184b4b/artifacts/best_d_model"
        pretrained_g_model_weights_path = "./mlruns/589683858730322811/ef2b79c7176d439d815f3e95c0184b4b/artifacts/best_g_model"
        pretrained_ema_g_model_weights_path = "./mlruns/589683858730322811/ef2b79c7176d439d815f3e95c0184b4b/artifacts/best_ema_g_model"
    else:
        print("Loading basic pretrained models...")
        pretrained_d_model_weights_path = "./results/pretrained_models/Real-ESRGAN/Discriminator_x2-DFO2K-e37ff529.pth.tar"
        pretrained_g_model_weights_path = "./results/pretrained_models/BSRGAN/BSRGAN_x2-DIV2K-62958d37.pth.tar"
        #pretrained_d_model_weights_path = ""
        #pretrained_g_model_weights_path = ""

    # Incremental training and migration training
    resume_d_model_weights_path = ""
    resume_g_model_weights_path = ""

    # Total num epochs (1,600,000 iters)
    epochs = 15
    print("Total Epochs -> "+str(epochs))

    # Feature extraction layer parameter configuration
    feature_model_extractor_nodes = ["features.2", "features.7", "features.16", "features.25", "features.34"]
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # Loss function weight
    #pixel_weight = [1.0]
    #pixel_weight = [60.0, 40.0, 30.0, 20.0, 10.0]
    pixel_weight = [20.0]
    #content_weight = [0.1, 0.1, 1.0, 1.0, 1.0]
    #content_weight = [0.1, 0.2, 0.5, 1.0]
    content_weight = [1.0]
    #adversarial_weight = [0.1]
    #adversarial_weight = [0.1, 0.2, 0.3, 0.5]
    adversarial_weight = [0.5]

    # Optimizer parameter
    #model_lr = 5e-5
    model_lr = 8e-5
    discriminator_lr = 2e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-4  # Keep no nan
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.999

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.5),int(epochs * 0.7)]
    #lr_scheduler_gamma = 0.5
    lr_scheduler_gamma = 0.85

    # How many iterations to print the training result
    train_print_frequency = 50
    valid_print_frequency = 200

if mode == "test":
    print("Test")
    print(f"Experiment: {experiment_id}\nRun Id: {run_id}")

    upscale_lpips_eval = 2

    save_images = True
    save_discriminator_eval = False
    save_metrics = True
    subdivision_lpips = False
    save_discriminator_attention_layers = False
    modelType = "best"

    print(f' save_images: {save_images}\n save_discriminator_eval: {save_discriminator_eval}\n save_metrics: {save_metrics}\n subdivision_lpips: {subdivision_lpips}\n save_discriminator_attention_layers: {save_discriminator_attention_layers}\n modelType: {modelType}\n')

    

    gt_dir = f"../data/AirfRANs/vtuNUT/test"
    #gt_dir = f"../data/Bubbles/test"
    #gt_dir = f"../data/DICOM/DICOM_Frame_Sequence"
    #gt_dir = f"../data/AmostraLR/Microscopio/0.5mm"

    #g_model_weights_path = f"./mlruns/"+experiment_id+"/"+run_id+"/artifacts/"+modelType+"_g_model"
    pretrained_g_model_weights_path = "./results/pretrained_models/BSRGAN/BSRGAN_x2-DIV2K-62958d37.pth.tar"
    

    if save_discriminator_eval or save_discriminator_attention_layers:
        d_model_weights_path = f"./mlruns/"+experiment_id+"/"+run_id+"/artifacts/"+modelType+"_d_model"
