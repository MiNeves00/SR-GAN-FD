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

import cv2
import torch
from natsort import natsorted

import bsrgan_config
import mlflow
import imgproc
import model
from image_quality_assessment import PSNR, SSIM, NIQE
from lpips import LPIPS
from utils import make_directory
from dataset import CUDAPrefetcher, TrainValidImageDataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision.utils import draw_segmentation_masks

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))



def main() -> None:

    # Set MLflow experiment & run
    mlflow.set_experiment(bsrgan_config.experience_name)
    try:
      mlflow.start_run(run_id=bsrgan_config.run_id, tags=bsrgan_config.tags, description=bsrgan_config.description)
    except: # If last session was not ended
      mlflow.end_run()
      mlflow.start_run(run_id=bsrgan_config.run_id, tags=bsrgan_config.tags, description=bsrgan_config.description)

    print("Continuing run with id:" + str(bsrgan_config.run_id))

    # Load Test Dataset
    test_datasets = TrainValidImageDataset(bsrgan_config.gt_dir,
                                        0,
                                        bsrgan_config.upscale_factor,
                                        "Valid",bsrgan_config.degradation_process_parameters_dict)
    
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)
    
    test_prefetcher = CUDAPrefetcher(test_dataloader, bsrgan_config.device)
    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()

    
    # Load Generator Model
    g_model = mlflow.pytorch.load_model(bsrgan_config.g_model_weights_path)
    bsrgan_model = g_model.to(device=bsrgan_config.device)
    bsrgan_model.eval()

    # Load Discriminator Model
    if bsrgan_config.save_discriminator_eval:
      d_model = mlflow.pytorch.load_model(bsrgan_config.d_model_weights_path)
      d_model = d_model.to(device=bsrgan_config.device)

      adversarial_criterion = nn.BCEWithLogitsLoss()
      adversarial_criterion = adversarial_criterion.to(device=bsrgan_config.device)
      d_model.eval()


    # Initialize the sharpness evaluation function
    psnr = PSNR(bsrgan_config.upscale_factor, bsrgan_config.only_test_y_channel)
    ssim = SSIM(bsrgan_config.upscale_factor, bsrgan_config.only_test_y_channel)
    niqe = NIQE(bsrgan_config.upscale_factor, bsrgan_config.niqe_model_path)
    lpips = LPIPS(net='alex')

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=bsrgan_config.device, non_blocking=True)
    ssim = ssim.to(device=bsrgan_config.device, non_blocking=True)
    niqe = niqe.to(device=bsrgan_config.device, non_blocking=True)
    lpips = lpips.to(device=bsrgan_config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0
    niqe_metrics = 0.0
    lpips_metrics = 0.0

    if bsrgan_config.save_discriminator_eval:
       sr_prob_metrics = 0.0
       gt_prob_metrics = 0.0
       gt_loss_metrics = 0.0
       sr_loss_metrics = 0.0
       total_loss_metrics = 0.0

    # Get a list of test image file names.
    file_names = os.listdir(bsrgan_config.gt_dir)
    # Get the number of test image files.
    total_files = int(len(file_names))

    pathLR = "testImagesLR/"
    pathTest = "testImages/"
    pathDiscriminatorGT = "testDiscriminatorGT/"
    pathDiscriminatorSR = "testDiscriminatorSR/"

    print("Starting tests...")

    for index in range(total_files):

        batch_data = test_prefetcher.next()
        gt_tensor = batch_data["gt"].to(device=bsrgan_config.device, non_blocking=True)
        lr_tensor = batch_data["lr"].to(device=bsrgan_config.device, non_blocking=True)

        if bsrgan_config.save_images:
          lr_image = imgproc.tensor_to_image(lr_tensor, False, False)
          #lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)
          mlflow.log_image(lr_image, pathLR+file_names[index])

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = bsrgan_model(lr_tensor)

        # Save image
        if bsrgan_config.save_images:
          sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
          #sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
          mlflow.log_image(sr_image, pathTest+file_names[index])
          #cv2.imwrite(sr_image_path, sr_image)

        if bsrgan_config.save_discriminator_eval:
          # Discriminator output, fake is 0 which is black. Real is 1 which is white
          batch_size, _, height, width = gt_tensor.shape
          real_label = torch.full([batch_size, 1, height, width], 1.0, dtype=gt_tensor.dtype, device=bsrgan_config.device)
          fake_label = torch.full([batch_size, 1, height, width], 0.0, dtype=gt_tensor.dtype, device=bsrgan_config.device)

          gt_output = d_model(gt_tensor)

          d_loss_hr = adversarial_criterion(gt_output, real_label)

          sr_output = d_model(sr_tensor.detach().clone())
          
          d_loss_sr = adversarial_criterion(sr_output, fake_label)

          d_loss = d_loss_hr + d_loss_sr

          d_gt_probability = torch.mean(torch.sigmoid_(gt_output.detach()))
          d_sr_probability = torch.mean(torch.sigmoid_(sr_output.detach()))

          gt_prob_metrics += d_gt_probability.item()
          sr_prob_metrics += d_sr_probability.item()
          gt_loss_metrics += d_loss_hr.item()
          sr_loss_metrics += d_loss_sr.item()
          total_loss_metrics += d_loss.item()



          gt_image = imgproc.tensor_to_image(torch.sigmoid_(gt_output), False, False)
          mlflow.log_image(gt_image, pathDiscriminatorGT+file_names[index])


          sr_image = imgproc.tensor_to_image(torch.sigmoid_(sr_output), False, False)
          mlflow.log_image(sr_image, pathDiscriminatorSR+file_names[index])

        # Cal IQA metrics
        psnr_metrics += psnr(sr_tensor, gt_tensor).item()
        ssim_metrics += ssim(sr_tensor, gt_tensor).item()
        niqe_metrics += niqe(sr_tensor).item()
        
        sr_tensor = 2*sr_tensor - 1 # Normalize from [0,1] to [-1,1]
        gt_tensor = 2*gt_tensor - 1

        # LPIPS in subimages
        if bsrgan_config.subdivision_lpips:
          subdivisions_upscale = bsrgan_config.upscale_lpips_eval

          img_sr_chunks_height = torch.chunk(sr_tensor, 2, dim=2)
          # split each of the two height chunks into two equal parts along the width dimension
          img_sr_subdivisions = []
          for chunk_height in img_sr_chunks_height:
              chunks_width = torch.chunk(chunk_height, 2, dim=3)
              img_sr_subdivisions.extend(chunks_width)


          img_gt_chunks_height = torch.chunk(gt_tensor, 2, dim=2)
          # split each of the two height chunks into two equal parts along the width dimension
          img_gt_subdivisions = []
          for chunk_height in img_gt_chunks_height:
              chunks_width = torch.chunk(chunk_height, 2, dim=3)
              img_gt_subdivisions.extend(chunks_width)

          lpips_total = lpips(sr_tensor, gt_tensor).item()
          #print(f'LPIPS Total image: {lpips_total}')

          avg_lpips_sub = 0
          sub_index = 0
          for subdivision_sr, subdivision_gt in zip(img_sr_subdivisions,img_gt_subdivisions):
            sr_image = imgproc.tensor_to_image(subdivision_sr, False, False)
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            mlflow.log_image(sr_image, "SubDivSR/"+str(sub_index)+"_"+file_names[index])
            gt_image = imgproc.tensor_to_image(subdivision_gt, False, False)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
            mlflow.log_image(gt_image, "SubDivGT/"+str(sub_index)+"_"+file_names[index])
            avg_lpips_sub += lpips(subdivision_sr, subdivision_gt).item()
            sub_index += 1

          avg_lpips_sub /= subdivisions_upscale*2
          #print(f'Avg LPIPS: {avg_lpips_sub}')
          lpips_metrics += avg_lpips_sub
        else:
           lpips_metrics += lpips(sr_tensor,gt_tensor).item()

        #if index == 4:
        #  break

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    # NIQE range value is 0~100 although it can go to infinite. Typically a score higher than 10 is bad and lower than 2 is excelent
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
    avg_niqe = 100 if niqe_metrics / total_files > 100 else niqe_metrics / total_files
    avg_lpips = 100 if lpips_metrics / total_files > 100 else lpips_metrics / total_files

    if bsrgan_config.save_discriminator_eval:
      avg_gt_prob_metrics = gt_prob_metrics / total_files
      avg_sr_prob_metrics = sr_prob_metrics / total_files
      avg_gt_loss_metrics = gt_loss_metrics / total_files
      avg_sr_loss_metrics = sr_loss_metrics / total_files
      avg_total_loss_metrics = total_loss_metrics / total_files
      print(f"GT Prob: {avg_gt_prob_metrics:4.4f} [dB]\n"
          f"SR Prob: {avg_sr_prob_metrics:4.4f} [u]\n"
          f"GT Loss: {avg_gt_loss_metrics:4.4f} [100u]\n"
          f"SR Loss: {avg_sr_loss_metrics:4.4f} [100u]\n"
          f"Total Loss: {avg_total_loss_metrics:4.4f} [100u]\n")

 
    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]\n"
          f"NIQE: {avg_niqe:4.2f} [100u]\n"
          f"LPIPS: {avg_lpips:4.2f} [100u]")

    metrics_dict = {"PSNR": avg_psnr, "SSIM": avg_ssim, "NIQE": avg_niqe, "LPIPS": avg_lpips}
    if bsrgan_config.save_discriminator_eval:
       metrics_dict = {"PSNR": avg_psnr, "SSIM": avg_ssim, "NIQE": avg_niqe, "LPIPS": avg_lpips, "GT Prob": avg_gt_prob_metrics, "SR Prob": avg_sr_prob_metrics, "GT Loss": avg_gt_loss_metrics, "SR Loss": avg_sr_loss_metrics, "Total Loss": avg_total_loss_metrics}

    if bsrgan_config.save_metrics:
      mlflow.log_dict(metrics_dict,"testMetrics.json")

    mlflow.end_run()


if __name__ == "__main__":
    main()

