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

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))



def main() -> None:

    # Set MLflow experiment & run
	mlflow.set_experiment(bsrgan_config.experience_name)

	try:
		mlflow.start_run(tags=bsrgan_config.tags, description=bsrgan_config.description)
	except: # If last session was not ended
		mlflow.end_run()
		mlflow.start_run(tags=bsrgan_config.tags, description=bsrgan_config.description)

	run = mlflow.active_run()
	print("Active run_id: {}".format(run.info.run_id))
	
	degrad=bsrgan_config.degradation_process_parameters_dict
	
	mlflow.log_params({'jpeg_prob':degrad["jpeg_prob"],'scale2_prob':degrad["scale2_prob"],'shuffle_prob':degrad["shuffle_prob"],'use_sharp':degrad["use_sharp"],})

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

    # Get a list of test image file names.
	file_names = os.listdir(bsrgan_config.gt_dir)
    # Get the number of test image files.
	total_files = int(len(file_names))

	pathLR = "testImagesLR/"

	print("Starting Degradations...")

	for index in range(total_files):

		batch_data = test_prefetcher.next()
		lr_tensor = batch_data["lr"].to(device=bsrgan_config.device, non_blocking=True)

		lr_image = imgproc.tensor_to_image(lr_tensor, False, False)
		lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)
		mlflow.log_image(lr_image, pathLR+file_names[index])
    
	print("Finished Degradations")

	mlflow.end_run()
	exit()




if __name__ == "__main__":
    main()

