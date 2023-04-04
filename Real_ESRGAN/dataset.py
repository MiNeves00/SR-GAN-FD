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
import math
import os
import queue
import random
import threading

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import imgproc

__all__ = [
    "DegeneratedImageDataset", "PairedImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class DegeneratedImageDataset(Dataset):
    """Define degenerate dataset loading method.

    Args:
        gt_images_dir (str): Ground-truth dataset address
        degradation_model_parameters_dict (dict): Parameter dictionary with degenerate model

    """

    def __init__(
            self,
            gt_images_dir: str,
            degradation_model_parameters_dict: dict
    ) -> None:
        super(DegeneratedImageDataset, self).__init__()
        # Get a list of all image filenames
        self.gt_image_file_names = [os.path.join(gt_images_dir, image_file_name) for image_file_name in
                                    os.listdir(gt_images_dir)]
        # Define the probability of each processing operation in the first-order degradation
        self.degradation_model_parameters_dict = degradation_model_parameters_dict
        # Define the size of the sinc filter kernel
        self.sinc_tensor = torch.zeros([degradation_model_parameters_dict["sinc_kernel_size"],
                                        degradation_model_parameters_dict["sinc_kernel_size"]]).float()
        self.sinc_tensor[degradation_model_parameters_dict["sinc_kernel_size"] // 2,
                         degradation_model_parameters_dict["sinc_kernel_size"] // 2] = 1

    def __getitem__(
            self,
            batch_index: int
    ) -> [Tensor, Tensor, Tensor] or [Tensor, Tensor]:
        # Generate a first-order degenerate Gaussian kernel
        gaussian_kernel_size1 = random.choice(self.degradation_model_parameters_dict["gaussian_kernel_range"])
        if np.random.uniform() < self.degradation_model_parameters_dict["sinc_kernel_probability1"]:
            # This sinc filter setting applies to kernels in the range [7, 21] and can be adjusted dynamically
            if gaussian_kernel_size1 < int(np.median(self.degradation_model_parameters_dict["gaussian_kernel_range"])):
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            gaussian_kernel1 = imgproc.generate_sinc_kernel(
                omega_c,
                gaussian_kernel_size1,
                padding=False)
        else:
            gaussian_kernel1 = imgproc.random_mixed_kernels(
                self.degradation_model_parameters_dict["gaussian_kernel_type"],
                self.degradation_model_parameters_dict["gaussian_kernel_probability1"],
                gaussian_kernel_size1,
                self.degradation_model_parameters_dict["gaussian_sigma_range1"],
                self.degradation_model_parameters_dict["gaussian_sigma_range1"],
                [-math.pi, math.pi],
                self.degradation_model_parameters_dict["generalized_kernel_beta_range1"],
                self.degradation_model_parameters_dict["plateau_kernel_beta_range1"],
                noise_range=None)
        # First-order degenerate Gaussian fill kernel size
        pad_size = (self.degradation_model_parameters_dict["gaussian_kernel_range"][-1] - gaussian_kernel_size1) // 2
        gaussian_kernel1 = np.pad(gaussian_kernel1, ((pad_size, pad_size), (pad_size, pad_size)))

        # Generate a second-order degenerate Gaussian kernel
        gaussian_kernel_size2 = random.choice(self.degradation_model_parameters_dict["gaussian_kernel_range"])
        if np.random.uniform() < self.degradation_model_parameters_dict["sinc_kernel_probability2"]:
            # This sinc filter setting applies to kernels in the range [7, 21] and can be adjusted dynamically
            if gaussian_kernel_size2 < int(np.median(self.degradation_model_parameters_dict["gaussian_kernel_range"])):
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            gaussian_kernel2 = imgproc.generate_sinc_kernel(
                omega_c,
                gaussian_kernel_size2,
                padding=False)
        else:
            gaussian_kernel2 = imgproc.random_mixed_kernels(
                self.degradation_model_parameters_dict["gaussian_kernel_type"],
                self.degradation_model_parameters_dict["gaussian_kernel_probability2"],
                gaussian_kernel_size2,
                self.degradation_model_parameters_dict["gaussian_sigma_range2"],
                self.degradation_model_parameters_dict["gaussian_sigma_range2"],
                [-math.pi, math.pi],
                self.degradation_model_parameters_dict["generalized_kernel_beta_range2"],
                self.degradation_model_parameters_dict["plateau_kernel_beta_range2"],
                noise_range=None)

        # second-order degenerate Gaussian fill kernel size
        pad_size = (self.degradation_model_parameters_dict["gaussian_kernel_range"][-1] - gaussian_kernel_size2) // 2
        gaussian_kernel2 = np.pad(gaussian_kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # Sinc filter kernel
        if np.random.uniform() < self.degradation_model_parameters_dict["sinc_kernel_probability3"]:
            gaussian_kernel_size2 = random.choice(self.degradation_model_parameters_dict["gaussian_kernel_range"])
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = imgproc.generate_sinc_kernel(
                omega_c,
                gaussian_kernel_size2,
                padding=self.degradation_model_parameters_dict["sinc_kernel_size"])
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.sinc_tensor

        gaussian_kernel1 = torch.FloatTensor(gaussian_kernel1)
        gaussian_kernel2 = torch.FloatTensor(gaussian_kernel2)
        sinc_kernel = torch.FloatTensor(sinc_kernel)

        # read a batch of images
        gt_image = cv2.imread(self.gt_image_file_names[batch_index])

        # BGR image data to RGB image data
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image data channel to a data format supported by PyTorch
        gt_tensor = imgproc.image_to_tensor(gt_image, False, False)

        return {"gt": gt_tensor,
                "gaussian_kernel1": gaussian_kernel1,
                "gaussian_kernel2": gaussian_kernel2,
                "sinc_kernel": sinc_kernel}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)


class PairedImageDataset(Dataset):
    """Image dataset loading method after registration

    Args:
        paired_gt_images_dir (str): ground truth images after registration
        paired_lr_images_dir (str): registered low-resolution images
    """

    def __init__(
            self,
            paired_gt_images_dir: str,
            paired_lr_images_dir: str,
    ) -> None:
        super(PairedImageDataset, self).__init__()
        # Get a list of all image filenames
        self.paired_gt_image_file_names = [os.path.join(paired_gt_images_dir, x) for x in
                                           os.listdir(paired_gt_images_dir)]
        self.paired_lr_image_file_names = [os.path.join(paired_lr_images_dir, x) for x in
                                           os.listdir(paired_lr_images_dir)]

    def __getitem__(self, batch_index: int) -> dict[str, Tensor]:
        # read a batch of images
        gt_image = cv2.imread(self.paired_gt_image_file_names[batch_index]).astype(np.float32) / 255.
        lr_image = cv2.imread(self.paired_lr_image_file_names[batch_index]).astype(np.float32) / 255.

        # BGR image data to RGB image data
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image data channel to a data format supported by PyTorch
        gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor, "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.paired_gt_image_file_names)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
