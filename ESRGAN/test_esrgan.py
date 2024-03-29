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

import esrgan_config
import mlflow
import imgproc
import model
from image_quality_assessment import PSNR, SSIM, NIQE
from lpips import LPIPS
from utils import make_directory
from dataset import CUDAPrefetcher, TrainValidImageDataset
from torch.utils.data import DataLoader

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))



def main() -> None:

    # Set MLflow experiment & run
    mlflow.set_experiment(esrgan_config.experience_name)
    try:
      mlflow.start_run(run_id=esrgan_config.run_id, tags=esrgan_config.tags, description=esrgan_config.description)
    except: # If last session was not ended
      mlflow.end_run()
      mlflow.start_run(run_id=esrgan_config.run_id, tags=esrgan_config.tags, description=esrgan_config.description)

    print("Continuing run with id:" + str(esrgan_config.run_id))

    # Load Test Dataset
    test_datasets = TrainValidImageDataset(esrgan_config.gt_dir,
                                        0,
                                        esrgan_config.upscale_factor,
                                        "Valid")
    
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)
    
    test_prefetcher = CUDAPrefetcher(test_dataloader, esrgan_config.device)
    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()


  '''
    # Initialize the super-resolution bsrgan_model
    esrgan_model = model.__dict__[esrgan_config.g_arch_name](in_channels=esrgan_config.in_channels,
                                                             out_channels=esrgan_config.out_channels,
                                                             channels=esrgan_config.channels,
                                                             growth_channels=esrgan_config.growth_channels,
                                                             num_blocks=esrgan_config.num_blocks)
    
    esrgan_model = esrgan_model.to(device=esrgan_config.device)
    print(f"Build `{esrgan_config.g_arch_name}` model successfully.")

    # Load the super-resolution bsrgan_model weights
    checkpoint = torch.load(esrgan_config.g_model_weights_path, map_location=lambda storage, loc: storage)
    esrgan_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{esrgan_config.g_arch_name}` model weights "
          f"`{os.path.abspath(esrgan_config.g_model_weights_path)}` successfully.")
    '''

    
    # Load Generator Model
    g_model = mlflow.pytorch.load_model(esrgan_config.g_model_weights_path)
    esrgan_model = g_model.to(device=esrgan_config.device)

    # Create a folder of super-resolution experiment results
    #make_directory(esrgan_config.sr_dir)

    # Start the verification mode of the bsrgan_model.
    esrgan_model.eval()

    # Initialize the sharpness evaluation function
    psnr = PSNR(esrgan_config.upscale_factor, esrgan_config.only_test_y_channel)
    ssim = SSIM(esrgan_config.upscale_factor, esrgan_config.only_test_y_channel)
    niqe = NIQE(esrgan_config.upscale_factor, esrgan_config.niqe_model_path)
    lpips = LPIPS(net='alex')

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=esrgan_config.device, non_blocking=True)
    ssim = ssim.to(device=esrgan_config.device, non_blocking=True)
    niqe = niqe.to(device=esrgan_config.device, non_blocking=True)
    lpips = lpips.to(device=esrgan_config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0
    niqe_metrics = 0.0
    lpips_metrics = 0.0

    # Get a list of test image file names.
    file_names = os.listdir(esrgan_config.gt_dir)
    # Get the number of test image files.
    total_files = int(len(file_names))

    pathLR = "testImagesLR/"
    pathTest = "testImages/"

    for index in range(total_files):
        '''
        lr_image_path = os.path.join(esrgan_config.lr_dir, file_names[index])
        sr_image_path = os.path.join(esrgan_config.sr_dir, file_names[index])
        gt_image_path = os.path.join(esrgan_config.gt_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, esrgan_config.device)
        gt_tensor = imgproc.preprocess_one_image(gt_image_path, esrgan_config.device)
        '''

        batch_data = test_prefetcher.next()
        gt_tensor = batch_data["gt"].to(device=esrgan_config.device, non_blocking=True)
        lr_tensor = batch_data["lr"].to(device=esrgan_config.device, non_blocking=True)

        lr_image = imgproc.tensor_to_image(lr_tensor, False, False)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)
        mlflow.log_image(lr_image, pathLR+file_names[index])

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = esrgan_model(lr_tensor)

        # Save image
        if esrgan_config.save_images:
          sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
          sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
          mlflow.log_image(sr_image, pathTest+file_names[index])
          #cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_tensor, gt_tensor).item()
        ssim_metrics += ssim(sr_tensor, gt_tensor).item()
        niqe_metrics += niqe(sr_tensor).item()
        
        sr_tensor = 2*sr_tensor - 1 # Normalize from [0,1] to [-1,1]
        gt_tensor = 2*gt_tensor - 1
        lpips_metrics += lpips(sr_tensor, gt_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    # NIQE range value is 0~100 although it can go to infinite. Typically a score higher than 10 is bad and lower than 2 is excelent
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
    avg_niqe = 100 if niqe_metrics / total_files > 100 else niqe_metrics / total_files
    avg_lpips = 100 if lpips_metrics / total_files > 100 else lpips_metrics / total_files
 
    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]\n"
          f"NIQE: {avg_niqe:4.2f} [100u]\n"
          f"LPIPS: {avg_lpips:4.2f} [100u]")

    metrics_dict = {"PSNR": avg_psnr, "SSIM": avg_ssim, "NIQE": avg_niqe, "LPIPS": avg_lpips}
    mlflow.log_dict(metrics_dict,"testMetrics.json")

    mlflow.end_run()


if __name__ == "__main__":
    main()

