# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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

import mlflow

import bsrgan_config
import model
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM, NIQE
from lpips import LPIPS
from imgproc import random_crop
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, valid_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    d_model, g_model, ema_g_model = build_model()
    print(f"Build `{bsrgan_config.g_model_arch_name}` model successfully.")

    pixel_criterion, content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")

    print("Check whether to load pretrained d model weights...")
    if bsrgan_config.pretrained_d_model_weights_path:
        d_model = load_state_dict(d_model, bsrgan_config.pretrained_d_model_weights_path)
        print(f"Loaded `{bsrgan_config.pretrained_d_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained d model weights not found.")

    print("Check whether to load pretrained g model weights...")
    if bsrgan_config.pretrained_g_model_weights_path:
        g_model = load_state_dict(g_model, bsrgan_config.pretrained_g_model_weights_path)
        print(f"Loaded `{bsrgan_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained g model weights not found.")

    d_optimizer, g_optimizer = define_optimizer(d_model, g_model)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    # Create a experiment results
    samples_dir = os.path.join("samples", bsrgan_config.exp_name)
    results_dir = os.path.join("results", bsrgan_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", bsrgan_config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(bsrgan_config.upscale_factor, bsrgan_config.only_test_y_channel)
    ssim_model = SSIM(bsrgan_config.upscale_factor, bsrgan_config.only_test_y_channel)
    niqe_model = NIQE(bsrgan_config.upscale_factor, bsrgan_config.niqe_model_path)
    lpips_model = LPIPS(net=bsrgan_config.lpips_net)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=bsrgan_config.device)
    ssim_model = ssim_model.to(device=bsrgan_config.device)
    niqe_model = niqe_model.to(device=bsrgan_config.device, non_blocking=True)
    lpips_model = lpips_model.to(device=bsrgan_config.device, non_blocking=True)

    best_lpips_metrics = 1.0


    # Start MLFlow Tracking
    try:
        mlflow.set_experiment(bsrgan_config.experience_name)
    except:
        experiment_id= mlflow.create_experiment(bsrgan_config.experience_name)
        print("New Experiment created with name: " + bsrgan_config.experience_name + " and ID: " + str(experiment_id))

    # Start MLflow run & log parameters 
    try:
      mlflow.start_run(run_name=bsrgan_config.run_name, tags=bsrgan_config.tags, description=bsrgan_config.description)
    except: # If last session was not ended
      mlflow.end_run()
      mlflow.start_run(run_name=bsrgan_config.run_name, tags=bsrgan_config.tags, description=bsrgan_config.description)
    
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))

    mlflow.log_params({'exp_name':bsrgan_config.exp_name,'d_arch_name':bsrgan_config.d_model_arch_name,'g_arch_name':bsrgan_config.g_model_arch_name,'d_in_channels':bsrgan_config.d_in_channels,'d_out_channels':bsrgan_config.d_out_channels,'d_channels':bsrgan_config.d_channels,'g_in_channels':bsrgan_config.g_in_channels,'g_out_channels':bsrgan_config.g_out_channels,'g_channels':bsrgan_config.g_channels,'growth_channels':bsrgan_config.g_growth_channels,'num_blocks':bsrgan_config.g_num_rrdb,'upscale_factor':bsrgan_config.upscale_factor,'gt_image_size':bsrgan_config.gt_image_size,'batch_size':bsrgan_config.batch_size,'train_gt_images_dir':bsrgan_config.train_gt_images_dir,'valid_gt_images_dir':bsrgan_config.valid_gt_images_dir,
                       'pretrained_d_model_weights_path':bsrgan_config.pretrained_d_model_weights_path,'pretrained_g_model_weights_path':bsrgan_config.pretrained_g_model_weights_path,'resume_d_model_weights_path':bsrgan_config.resume_d_model_weights_path,'resume_g_model_weights_path':bsrgan_config.resume_g_model_weights_path,'epochs':bsrgan_config.epochs,'pixel_weight':bsrgan_config.pixel_weight,'content_weight':bsrgan_config.content_weight,'adversarial_weight':bsrgan_config.adversarial_weight,'feature_model_extractor_nodes':bsrgan_config.feature_model_extractor_nodes,'feature_model_normalize_mean':bsrgan_config.feature_model_normalize_mean,'feature_model_normalize_std':bsrgan_config.feature_model_normalize_std,'model_lr':bsrgan_config.model_lr,'model_betas':bsrgan_config.model_betas,'model_eps':bsrgan_config.model_eps,'model_weight_decay':bsrgan_config.model_weight_decay,'model_ema_decay':bsrgan_config.model_ema_decay,'lr_scheduler_milestones':bsrgan_config.lr_scheduler_milestones,'lr_scheduler_gamma':bsrgan_config.lr_scheduler_gamma,'lpips_net':bsrgan_config.lpips_net,'niqe_model_path':bsrgan_config.niqe_model_path})


    for epoch in range(start_epoch, bsrgan_config.epochs):
        pixel_loss, content_loss, adversarial_loss, d_gt_probabilities, d_sr_probabilities = train(d_model,
              g_model,
              ema_g_model,
              train_prefetcher,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler,
              writer,
              bsrgan_config.device,
              bsrgan_config.train_print_frequency)
        psnr_val, ssim_val, niqe_val, lpips_val = validate(g_model,
                              valid_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              niqe_model,
                              lpips_model,
                              "Valid")
        print("\n")

        log_epoch(pixel_loss, content_loss, adversarial_loss, d_gt_probabilities, d_sr_probabilities, psnr_val, ssim_val, niqe_val, lpips_val, epoch)

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Save the best model with the highest LPIPS score in validation dataset
        is_best = lpips_val < best_lpips_metrics
        best_lpips_metrics = min(lpips_val, best_lpips_metrics)

        if is_best:
          print("Saving best model...")
          mlflow.pytorch.log_model(g_model, "g_model")
          mlflow.pytorch.log_model(d_model, "d_model")
          print("Finished Saving")
        else:
          print("Was not the best")

    # End logging
    mlflow.end_run()

def log_epoch(g_pixel_loss, g_content_loss, g_adversarial_loss, d_gt_probabilities, d_sr_probabilities, psnr_val, ssim_val, niqe_val, lpips_val, epoch):
    '''
    g_pixel_loss, g_content_loss, g_adversarial_loss: train generator loss
    d_gt_probabilities, d_sr_probabilities: descriminator probabilities
    psnr, ssim, niqe, lpips: validation metrics
    '''

    print('\nLogging epoch data...')

    g_train_loss = g_pixel_loss + g_content_loss + g_adversarial_loss

    mlflow.log_metrics({'g_train_loss':g_train_loss, 'g_pixel_loss':g_pixel_loss, 'g_content_loss':g_content_loss, 'g_adversarial_loss':g_adversarial_loss, 'd_gt_probabilities':d_gt_probabilities, 'd_sr_probabilities':d_sr_probabilities, 'psnr_val':psnr_val, 'ssim_val':ssim_val, 'niqe_val':niqe_val, 'lpips_val':lpips_val}, step=epoch)

    print('Finished Logging\n')


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(bsrgan_config.train_gt_images_dir,
                                            bsrgan_config.crop_image_size,
                                            bsrgan_config.upscale_factor,
                                            "Train",
                                            bsrgan_config.degradation_process_parameters_dict)
    '''test_datasets = TestImageDataset(bsrgan_config.test_gt_images_dir, bsrgan_config.test_lr_images_dir)'''

    valid_datasets = TrainValidImageDataset(bsrgan_config.valid_gt_images_dir,
                                            bsrgan_config.crop_image_size,
                                            bsrgan_config.upscale_factor,
                                            "Valid",
                                            bsrgan_config.degradation_process_parameters_dict)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=bsrgan_config.batch_size,
                                  shuffle=True,
                                  num_workers=bsrgan_config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                 batch_size=bsrgan_config.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, bsrgan_config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, bsrgan_config.device)

    return train_prefetcher, valid_prefetcher


def build_model() -> [nn.Module, nn.Module, nn.Module]:
    d_model = model.__dict__[bsrgan_config.d_model_arch_name](
        in_channels=bsrgan_config.d_in_channels,
        out_channels=bsrgan_config.d_out_channels,
        channels=bsrgan_config.d_channels,
    )
    g_model = model.__dict__[bsrgan_config.g_model_arch_name](
        in_channels=bsrgan_config.g_in_channels,
        out_channels=bsrgan_config.g_out_channels,
        channels=bsrgan_config.g_channels,
        growth_channels=bsrgan_config.g_growth_channels,
        num_rrdb=bsrgan_config.g_num_rrdb,
    )
    d_model = d_model.to(device=bsrgan_config.device)
    g_model = g_model.to(device=bsrgan_config.device)

    # Create an Exponential Moving Average Model
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - bsrgan_config.model_ema_decay) * averaged_model_parameter + bsrgan_config.model_ema_decay * model_parameter
    ema_g_model = AveragedModel(g_model, avg_fn=ema_avg)

    return d_model, g_model, ema_g_model


def define_loss() -> [nn.L1Loss, model.ContentLoss, nn.BCEWithLogitsLoss]:
    pixel_criterion = nn.L1Loss()
    content_criterion = model.ContentLoss(bsrgan_config.feature_model_extractor_nodes,
                                          bsrgan_config.feature_model_normalize_mean,
                                          bsrgan_config.feature_model_normalize_std)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device=bsrgan_config.device)
    content_criterion = content_criterion.to(device=bsrgan_config.device)
    adversarial_criterion = adversarial_criterion.to(device=bsrgan_config.device)

    return pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(d_model, g_model) -> [optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(d_model.parameters(),
                             bsrgan_config.model_lr,
                             bsrgan_config.model_betas,
                             bsrgan_config.model_eps,
                             bsrgan_config.model_weight_decay)
    g_optimizer = optim.Adam(g_model.parameters(),
                             bsrgan_config.model_lr,
                             bsrgan_config.model_betas,
                             bsrgan_config.model_eps,
                             bsrgan_config.model_weight_decay)

    return d_optimizer, g_optimizer


def define_scheduler(
        d_optimizer: optim.Adam,
        g_optimizer: optim.Adam
) -> [lr_scheduler.MultiStepLR, lr_scheduler.MultiStepLR]:
    d_scheduler = lr_scheduler.MultiStepLR(d_optimizer,
                                           bsrgan_config.lr_scheduler_milestones,
                                           bsrgan_config.lr_scheduler_gamma)
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer,
                                           bsrgan_config.lr_scheduler_milestones,
                                           bsrgan_config.lr_scheduler_gamma)
    return d_scheduler, g_scheduler


def train(
        d_model: nn.Module,
        g_model: nn.Module,
        ema_g_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        content_criterion: model.content_loss,
        adversarial_criterion: nn.BCEWithLogitsLoss,
        d_optimizer: optim.Adam,
        g_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
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

    # Put the generative network model in training mode
    d_model.train()
    g_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    #limit = 12

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=device, non_blocking=True)
        lr = batch_data["lr"].to(device=device, non_blocking=True)
        pixel_weight = torch.Tensor(bsrgan_config.pixel_weight).to(device=device)
        content_weight = torch.Tensor(bsrgan_config.content_weight).to(device=device)
        adversarial_weight = torch.Tensor(bsrgan_config.adversarial_weight).to(device=device)

        # Crop image patch
        gt, lr = random_crop(gt, lr, bsrgan_config.gt_image_size, bsrgan_config.upscale_factor)

        # Set the real sample label to 1, and the false sample label to 0
        batch_size, _, height, width = gt.shape
        real_label = torch.full([batch_size, 1, height, width], 1.0, dtype=gt.dtype, device=device)
        fake_label = torch.full([batch_size, 1, height, width], 0.0, dtype=gt.dtype, device=device)

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        with amp.autocast():
            gt_output = d_model(gt)
            d_loss_hr = adversarial_criterion(gt_output, real_label)
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        scaler.scale(d_loss_hr).backward(retain_graph=True)

        # Calculate the classification score of the discriminator model for fake samples
        with amp.autocast():
            # Use the generator model to generate fake samples
            sr = g_model(lr)
            sr_output = d_model(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
        # Call the gradient scaling function in the mixed precision API
        # to back-propagate the gradient information of the fake samples
        scaler.scale(d_loss_sr).backward()

        # Calculate the total discriminator loss value
        d_loss = d_loss_hr + d_loss_sr

        # Improve the discriminator model's ability to classify real and fake samples
        scaler.step(d_optimizer)
        scaler.update()
        # Finish training the discriminator model

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        with amp.autocast():
            pixel_loss = pixel_criterion(sr, gt)
            content_loss = content_criterion(sr, gt)
            adversarial_loss = adversarial_criterion(d_model(sr), real_label)
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))
            content_loss = torch.sum(torch.mul(content_weight, content_loss))
            adversarial_loss = torch.sum(torch.mul(adversarial_weight, adversarial_loss))
            # Calculate the generator total loss value
            g_loss = pixel_loss + content_loss + adversarial_loss
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        scaler.scale(g_loss).backward()
        # Encourage the generator to generate higher quality fake samples,
        # making it easier to fool the discriminator
        scaler.step(g_optimizer)
        scaler.update()

        # Update EMA
        ema_g_model.update_parameters(g_model)
        # Finish training the generator model

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
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the
        # terminal print data normally
        batch_index += 1

        '''if batch_index>limit:
          print('Batch limit reached')
          return pixel_losses.avg, content_losses.avg, adversarial_losses.avg, d_gt_probabilities.avg, d_sr_probabilities.avg'''
    
    return pixel_losses.avg, content_losses.avg, adversarial_losses.avg, d_gt_probabilities.avg, d_sr_probabilities.avg


def validate(
        bsrnet_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        niqe_model: nn.Module,
        lpips_model: nn.Module,
        mode: str = "Valid",
) -> [float, float]:
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    niqees = AverageMeter("NIQE", ":4.2f")
    lpipses = AverageMeter("LPIPS", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes, niqees, lpipses], prefix=f"{mode}: ")

    print_freq = 1
    if mode == "Valid":
      print_freq = bsrgan_config.valid_print_frequency
    else:
      print_freq = bsrgan_config.test_print_frequency

    # Set the model as validation model
    bsrnet_model.eval()

    # Initialize data batches
    batch_index = 0

    #limit = 20

    # Set the data set iterator pointer to 0 and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Record the start time of verifying a batch
    end = time.time()

    # Disable gradient propagation
    with torch.no_grad():
        while batch_data is not None:
            # Load batches of data
            gt = batch_data["gt"].to(device=bsrgan_config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=bsrgan_config.device, non_blocking=True)

            # inference
            sr = bsrnet_model(lr)

            # Calculate the image IQA
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            niqe = niqe_model(sr)
            sr_tensor = 2*sr - 1 # Normalize from [0,1] to [-1,1]
            gt_tensor = 2*gt - 1
            lpips = lpips_model(sr, gt)

            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))
            niqees.update(niqe.item(), lr.size(0))
            lpipses.update(lpips.item(), lr.size(0))

            # Record the total time to verify a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output a verification log information
            if batch_index % print_freq == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # Add 1 to the number of data batches
            batch_index += 1

            '''if batch_index > limit:
              print("Limit reached")
              break'''

    # Print the performance index of the model at the current epoch
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
        writer.add_scalar(f"{mode}/NIQE", niqees.avg, epoch + 1)
        writer.add_scalar(f"{mode}/LPIPS", lpipses.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg, niqees.avg, lpipses.avg


if __name__ == "__main__":
    main()
