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
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import imgproc
import model
import realesrgan_config
from dataset import CUDAPrefetcher, DegeneratedImageDataset, PairedImageDataset
from image_quality_assessment import NIQE
from utils import load_state_dict, make_directory, save_checkpoint, validate, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    degenerated_train_prefetcher, paired_test_prefetcher = load_dataset()
    print("Load dataset successfully.")

    d_model, g_model, ema_g_model = build_model()
    print(f"Build `{realesrgan_config.d_model_arch_name}` model "
          f"`{realesrgan_config.g_model_arch_name}` model successfully.")

    pixel_criterion, content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_optimizer(d_model, g_model)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    # Load the pre-trained model weights and fine-tune the model
    print("Check whether to load pretrained d model weights...")
    if realesrgan_config.pretrained_d_model_weights_path:
        #d_model = mlflow.pytorch.load_model(realesrgan_config.pretrained_d_model_weights_path)
        d_model = load_state_dict(d_model, realesrgan_config.pretrained_d_model_weights_path)
        print(f"Loaded `{realesrgan_config.pretrained_d_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained d model weights not found.")
    print("Check whether to load pretrained g model weights...")
    if realesrgan_config.pretrained_g_model_weights_path:
        #g_model = mlflow.pytorch.load_model(realesrgan_config.pretrained_g_model_weights_path)
        g_model = load_state_dict(g_model, realesrgan_config.pretrained_g_model_weights_path)
        print(f"Loaded `{realesrgan_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained g model weights not found.")

    # Load the last training interruption node
    print("Check whether the resumed model is restored...")
    if realesrgan_config.resume_d_model_weights_path:
        g_model, _, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            d_model,
            realesrgan_config.resume_d_model_weights_path,
            None,
            d_optimizer,
            d_scheduler,
            "resume")
        print("Loaded resume d model weights.")
    else:
        print("Resume training d model not found. Start training from scratch.")
    print("Check whether the resume g model is restored...")
    if realesrgan_config.resume_g_model_weights_path:
        g_model, ema_g_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            g_model,
            realesrgan_config.resume_g_model_weights_path,
            ema_g_model,
            g_optimizer,
            g_scheduler,
            "resume")
        print("Loaded resume g model weights.")
    else:
        print("Resume training g model not found. Start training from scratch.")

    # Model weight training save address
    samples_dir = os.path.join("samples", realesrgan_config.exp_name)
    results_dir = os.path.join("results", realesrgan_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", realesrgan_config.exp_name))

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Initialize the image clarity evaluation method
    niqe_model = NIQE(realesrgan_config.upscale_factor, realesrgan_config.niqe_model_path)
    niqe_model = niqe_model.to(device=realesrgan_config.device)

    # Initialize training to generate network evaluation indicators
    best_niqe = 100.0

    for epoch in range(start_epoch, realesrgan_config.epochs):
        train(d_model,
              g_model,
              ema_g_model,
              degenerated_train_prefetcher,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler,
              writer,
              realesrgan_config.device,
              realesrgan_config.train_print_frequency)
        niqe = validate(g_model,
                        paired_test_prefetcher,
                        epoch,
                        writer,
                        niqe_model,
                        realesrgan_config.device,
                        realesrgan_config.test_print_frequency,
                        "Test")
        print("\n")

        # Update the learning rate after each training epoch
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save model weights
        is_best = niqe < best_niqe
        is_last = (epoch + 1) == realesrgan_config.epochs
        best_niqe = min(niqe, best_niqe)
        save_checkpoint({"epoch": epoch + 1,
                         "best_niqe": best_niqe,
                         "state_dict": d_model.state_dict(),
                         "optimizer": d_optimizer.state_dict(),
                         "scheduler": d_scheduler.state_dict()},
                        f"d_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_best.pth.tar",
                        "d_last.pth.tar",
                        is_best,
                        is_last)

        save_checkpoint({"epoch": epoch + 1,
                         "best_niqe": best_niqe,
                         "state_dict": g_model.state_dict(),
                         "ema_state_dict": ema_g_model.state_dict(),
                         "optimizer": g_optimizer.state_dict(),
                         "scheduler": g_scheduler.state_dict()},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    """Load training dataset"""
    train_datasets = DegeneratedImageDataset(realesrgan_config.train_gt_images_dir,
                                                         realesrgan_config.degradation_model_parameters_dict)
    valid_datasets = DegeneratedImageDataset(realesrgan_config.test_gt_images_dir,
                                                         realesrgan_config.degradation_model_parameters_dict)
    # generate dataset iterator
    train_dataloader = DataLoader(train_datasets,
                                              batch_size=realesrgan_config.batch_size,
                                              shuffle=True,
                                              num_workers=realesrgan_config.num_workers,
                                              pin_memory=True,
                                              drop_last=True,
                                              persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=True,
                                        drop_last=False,
                                        persistent_workers=True)

    # Replace the data set iterator with CUDA to speed up
    train_prefetcher = CUDAPrefetcher(train_dataloader, realesrgan_config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, realesrgan_config.device)

    return train_prefetcher, valid_prefetcher


def build_model() -> [nn.Module, nn.Module, nn.Module]:
    """Initialize the model"""
    d_model = model.__dict__[realesrgan_config.d_model_arch_name](
        in_channels=realesrgan_config.d_in_channels,
        out_channels=realesrgan_config.d_out_channels,
        channels=realesrgan_config.d_channels,
    )
    g_model = model.__dict__[realesrgan_config.g_model_arch_name](
        in_channels=realesrgan_config.g_in_channels,
        out_channels=realesrgan_config.g_out_channels,
        channels=realesrgan_config.g_channels,
        growth_channels=realesrgan_config.g_growth_channels,
        num_rrdb=realesrgan_config.g_num_rrdb,
    )
    d_model = d_model.to(device=realesrgan_config.device)
    g_model = g_model.to(device=realesrgan_config.device)

    # Generate an exponential average model based on the generator to stabilize model training
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - realesrgan_config.model_ema_decay) * averaged_model_parameter + realesrgan_config.model_ema_decay * model_parameter
    ema_g_model = AveragedModel(g_model, avg_fn=ema_avg)

    return d_model, g_model, ema_g_model


def define_loss() -> [nn.L1Loss, model.ContentLoss, nn.BCEWithLogitsLoss]:
    pixel_criterion = nn.L1Loss()
    content_criterion = model.ContentLoss(realesrgan_config.feature_model_extractor_nodes,
                                          realesrgan_config.feature_model_normalize_mean,
                                          realesrgan_config.feature_model_normalize_std)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device=realesrgan_config.device)
    content_criterion = content_criterion.to(device=realesrgan_config.device)
    adversarial_criterion = adversarial_criterion.to(device=realesrgan_config.device)

    return pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(d_model: nn.Module, g_model: nn.Module) -> [optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(d_model.parameters(),
                             realesrgan_config.model_lr,
                             realesrgan_config.model_betas,
                             realesrgan_config.model_eps,
                             realesrgan_config.model_weight_decay)
    g_optimizer = optim.Adam(g_model.parameters(),
                             realesrgan_config.model_lr,
                             realesrgan_config.model_betas,
                             realesrgan_config.model_eps,
                             realesrgan_config.model_weight_decay)

    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> [lr_scheduler.MultiStepLR,
                                                                           lr_scheduler.MultiStepLR]:
    d_scheduler = lr_scheduler.MultiStepLR(d_optimizer,
                                           realesrgan_config.lr_scheduler_milestones,
                                           realesrgan_config.lr_scheduler_gamma)
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer,
                                           realesrgan_config.lr_scheduler_milestones,
                                           realesrgan_config.lr_scheduler_gamma)

    return d_scheduler, g_scheduler


def train(
        d_model: nn.Module,
        g_model: nn.Module,
        ema_g_model: nn.Module,
        degenerated_train_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        feature_criterion: model.ContentLoss,
        adversarial_criterion: nn.BCEWithLogitsLoss,
        d_optimizer: optim.Adam,
        g_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> None:
    """training main function

    Args:
        d_model (nn.Module): discriminator model
        g_model (nn.Module): generator model
        ema_g_model (nn.Module): Generator-based exponential mean model
        degenerated_train_prefetcher (CUDARefetcher): training dataset iterator
        pixel_criterion (nn.L1Loss): pixel loss function
        feature_criterion (model.FeatureLoss): feature loss function
        adversarial_criterion (nn.BCEWithLogitsLoss): Adversarial loss function
        d_optimizer (optim.Adam): Discriminator model optimizer function
        g_optimizer (optim.Adam): generator model optimizer function
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
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_gt_probabilities = AverageMeter("D(GT)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses,
                              d_gt_probabilities, d_sr_probabilities],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put all model in train mode.
    d_model.train()
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

        # Load batches of data
        gt = batch_data["gt"].to(device=device, non_blocking=True)
        gaussian_kernel1 = batch_data["gaussian_kernel1"].to(device=device, non_blocking=True)
        gaussian_kernel2 = batch_data["gaussian_kernel2"].to(device=device, non_blocking=True)
        sinc_kernel = batch_data["sinc_kernel"].to(device=device, non_blocking=True)
        pixel_weight = torch.Tensor(realesrgan_config.pixel_weight).to(device=device)
        content_weight = torch.Tensor(realesrgan_config.content_weight).to(device=device)
        adversarial_weight = torch.Tensor(realesrgan_config.adversarial_weight).to(device=device)

        # Get the degraded low-resolution image
        gt_usm, gt, lr = imgproc.degradation_process(gt,
                                                     gaussian_kernel1,
                                                     gaussian_kernel2,
                                                     sinc_kernel,
                                                     realesrgan_config.upscale_factor,
                                                     realesrgan_config.degradation_process_parameters_dict,
                                                     jpeg_operation,
                                                     usm_sharpener)

        # image data augmentation
        (gt_usm, gt), lr = imgproc.random_crop_torch([gt_usm, gt], lr, realesrgan_config.gt_image_size, realesrgan_config.upscale_factor)
        (gt_usm, gt), lr = imgproc.random_rotate_torch([gt_usm, gt], lr, realesrgan_config.upscale_factor, [0, 90, 180, 270])
        (gt_usm, gt), lr = imgproc.random_vertically_flip_torch([gt_usm, gt], lr)
        (gt_usm, gt), lr = imgproc.random_horizontally_flip_torch([gt_usm, gt], lr)

        # Set the real sample label to 1, and the false sample label to 0
        batch_size, _, height, width = gt.shape
        real_label = torch.full([batch_size, 1, height, width], 1.0, dtype=torch.float, device=device)
        fake_label = torch.full([batch_size, 1, height, width], 0.0, dtype=torch.float, device=device)

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        with amp.autocast():
            # Use the generator model to generate fake samples
            sr = g_model(lr)
            pixel_loss = pixel_criterion(sr, gt_usm)
            feature_loss = feature_criterion(sr, gt_usm)
            adversarial_loss = adversarial_criterion(d_model(sr), real_label)
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))
            content_loss = torch.sum(torch.mul(content_weight, feature_loss))
            adversarial_loss = torch.sum(torch.mul(adversarial_weight, adversarial_loss))
            # Calculate the generator total loss value
            g_loss = pixel_loss + content_loss + adversarial_loss
        # Call the gradient scaling function in the mixed precision API to
        # bp the gradient information of the fake samples
        scaler.scale(g_loss).backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        scaler.step(g_optimizer)
        scaler.update()
        # Finish training the generator model

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        with amp.autocast():
            gt_output = d_model(gt)
            d_loss_gt = adversarial_criterion(gt_output, real_label)
        # Call the gradient scaling function in the mixed precision API to
        # bp the gradient information of the fake samples
        scaler.scale(d_loss_gt).backward()

        # Calculate the classification score of the discriminator model for fake samples
        with amp.autocast():
            sr_output = d_model(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
            # Calculate the total discriminator loss value
            d_loss = d_loss_sr + d_loss_gt
        # Call the gradient scaling function in the mixed precision API to
        # bp the gradient information of the fake samples
        scaler.scale(d_loss_sr).backward()
        # Improve the discriminator model's ability to classify real and fake samples
        scaler.step(d_optimizer)
        scaler.update()
        # Finish training the discriminator model

        # update exponential average model weights
        ema_g_model.update_parameters(g_model)

        # Calculate the score of the discriminator on real samples and fake samples,
        # the score of real samples is close to 1, and the score of fake samples is close to 0
        d_gt_probability = torch.sigmoid_(torch.mean(gt_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))

        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_gt_probabilities.update(d_gt_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", d_gt_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = degenerated_train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the
        # terminal print data normally
        batch_index += 1


if __name__ == "__main__":
    main()
