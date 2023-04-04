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
import os
import time

import torch
from torch import nn, optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import imgproc
import model
import realesrnet_config
from dataset import CUDAPrefetcher, DegeneratedImageDataset, PairedImageDataset
from image_quality_assessment import NIQE
from utils import load_state_dict, make_directory, save_checkpoint, validate, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_niqe = 100.0

    degenerated_train_prefetcher, paired_test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    g_model, ema_g_model = build_model()
    print(f"Build `{realesrnet_config.g_model_arch_name}` model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(g_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler successfully.")

    # Load the pre-trained model weights and fine-tune the model
    print("Check whether to load pretrained model weights...")
    if realesrnet_config.pretrained_g_model_weights_path:
        g_model = load_state_dict(g_model, realesrnet_config.pretrained_g_model_weights_path)
        print(f"Loaded `{realesrnet_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # Load the last training interruption node
    print("Check whether the resume model is restored...")
    if realesrnet_config.resume_g_model_weights_path:
        g_model, ema_g_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            g_model,
            realesrnet_config.pretrained_g_model_weights_path,
            ema_g_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded resume model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Model weight save address
    samples_dir = os.path.join("samples", realesrnet_config.exp_name)
    results_dir = os.path.join("results", realesrnet_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", realesrnet_config.exp_name))

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Initialize the image clarity evaluation method
    niqe_model = NIQE(realesrnet_config.upscale_factor, realesrnet_config.niqe_model_path)
    niqe_model = niqe_model.to(device=realesrnet_config.device)

    for epoch in range(start_epoch, realesrnet_config.epochs):
        train(g_model,
              ema_g_model,
              degenerated_train_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              realesrnet_config.device,
              realesrnet_config.train_print_frequency)
        niqe = validate(g_model,
                        paired_test_prefetcher,
                        epoch,
                        writer,
                        niqe_model,
                        realesrnet_config.device,
                        realesrnet_config.test_print_frequency,
                        "Test")
        print("\n")

        # Update the learning rate after each training epoch
        scheduler.step()

        # Automatically save model weights
        is_best = niqe < best_niqe
        is_last = (epoch + 1) == realesrnet_config.epochs
        best_niqe = min(niqe, best_niqe)
        save_checkpoint({"epoch": epoch + 1,
                         "best_niqe": best_niqe,
                         "state_dict": g_model.state_dict(),
                         "ema_state_dict": ema_g_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    """Load training dataset"""
    degenerated_train_datasets = DegeneratedImageDataset(realesrnet_config.degradation_train_gt_images_dir,
                                                         realesrnet_config.degradation_model_parameters_dict)
    paired_test_datasets = PairedImageDataset(realesrnet_config.degradation_test_gt_images_dir,
                                              realesrnet_config.degradation_test_lr_images_dir)
    # generate dataset iterator
    degenerated_train_dataloader = DataLoader(degenerated_train_datasets,
                                              batch_size=realesrnet_config.batch_size,
                                              shuffle=True,
                                              num_workers=realesrnet_config.num_workers,
                                              pin_memory=True,
                                              drop_last=True,
                                              persistent_workers=True)
    paired_test_dataloader = DataLoader(paired_test_datasets,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=True,
                                        drop_last=False,
                                        persistent_workers=True)

    # Replace the data set iterator with CUDA to speed up
    degenerated_train_prefetcher = CUDAPrefetcher(degenerated_train_dataloader, realesrnet_config.device)
    paired_test_prefetcher = CUDAPrefetcher(paired_test_dataloader, realesrnet_config.device)

    return degenerated_train_prefetcher, paired_test_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    """Initialize the model"""
    g_model = model.__dict__[realesrnet_config.g_model_arch_name](in_channels=realesrnet_config.g_in_channels,
                                                                  out_channels=realesrnet_config.g_out_channels,
                                                                  channels=realesrnet_config.g_channels,
                                                                  growth_channels=realesrnet_config.g_growth_channels,
                                                                  num_rrdb=realesrnet_config.g_num_rrdb)
    g_model = g_model.to(device=realesrnet_config.device)

    # Generate an exponential average model based on the generator to stabilize model training
    ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - realesrnet_config.model_ema_decay) * averaged_model_parameter + realesrnet_config.model_ema_decay * model_parameter
    ema_g_model = AveragedModel(g_model, avg_fn=ema_avg_fn)

    return g_model, ema_g_model


def define_loss() -> nn.L1Loss:
    pixel_criterion = nn.L1Loss()
    pixel_criterion = pixel_criterion.to(device=realesrnet_config.device)

    return pixel_criterion


def define_optimizer(g_model: nn.Module) -> optim.Adam:
    optimizer = optim.Adam(g_model.parameters(),
                           realesrnet_config.model_lr,
                           realesrnet_config.model_betas,
                           realesrnet_config.model_eps,
                           realesrnet_config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer: optim.Adam) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer,
                                    realesrnet_config.lr_scheduler_step_size,
                                    realesrnet_config.lr_scheduler_gamma)

    return scheduler


def train(
        g_model: nn.Module,
        ema_g_model: nn.Module,
        degenerated_train_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> None:
    """training main function

    Args:
        g_model (nn.Module): generator model
        ema_g_model (nn.Module): Generator-based exponential mean model
        degenerated_train_prefetcher (CUDARefetcher): training dataset iterator
        criterion (nn.L1Loss): loss function
        optimizer (optim.Adam): optimizer function
        epoch (int): number of training epochs
        scaler (amp.GradScaler): mixed precision function
        writer (SummaryWriter): training log function
        device (torch.device): The model of the evaluation model running device. Default: ``torch.device("cpu")``
        print_frequency (int): how many times to output the indicator. Default: 1

    """
    # Define JPEG compression method and USM sharpening method
    jpeg_operation = imgproc.DiffJPEG()
    usm_sharpener = imgproc.USMSharp()
    jpeg_operation = jpeg_operation.to(device=device)
    usm_sharpener = usm_sharpener.to(device=device)

    # Calculate how many batches of data there are under a dataset iterator
    batches = len(degenerated_train_prefetcher)
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in training mode
    g_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    degenerated_train_prefetcher.reset()
    batch_data = degenerated_train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        gt = batch_data["gt"].to(device=device, non_blocking=True)
        gaussian_kernel1 = batch_data["gaussian_kernel1"].to(device=device, non_blocking=True)
        gaussian_kernel2 = batch_data["gaussian_kernel2"].to(device=device, non_blocking=True)
        sinc_kernel = batch_data["sinc_kernel"].to(device=device, non_blocking=True)
        loss_weight = torch.Tensor(realesrnet_config.loss_weight).to(device=device)

        # Get the degraded low-resolution image
        gt_usm, gt, lr = imgproc.degradation_process(gt,
                                                     gaussian_kernel1,
                                                     gaussian_kernel2,
                                                     sinc_kernel,
                                                     realesrnet_config.upscale_factor,
                                                     realesrnet_config.degradation_process_parameters_dict,
                                                     jpeg_operation,
                                                     usm_sharpener)

        # image data augmentation
        (gt_usm, gt), lr = imgproc.random_crop_torch([gt_usm, gt], lr, realesrnet_config.gt_image_size,
                                                     realesrnet_config.upscale_factor)
        (gt_usm, gt), lr = imgproc.random_rotate_torch([gt_usm, gt], lr, realesrnet_config.upscale_factor,
                                                       [0, 90, 180, 270])
        (gt_usm, gt), lr = imgproc.random_vertically_flip_torch([gt_usm, gt], lr)
        (gt_usm, gt), lr = imgproc.random_horizontally_flip_torch([gt_usm, gt], lr)

        # Initialize the generator gradient
        g_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = g_model(lr)
            loss = criterion(sr, gt_usm)
            loss = torch.sum(torch.mul(loss_weight, loss))

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # update exponential average model weights
        ema_g_model.update_parameters(g_model)

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record training log information
        if batch_index % print_frequency == 0:
            # Writer Loss to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = degenerated_train_prefetcher.next()

        # Add 1 to the number of data batches
        batch_index += 1


if __name__ == "__main__":
    main()
