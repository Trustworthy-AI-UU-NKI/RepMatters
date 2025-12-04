""" Class to define a LightningDataModule for the PolypGen2021 dataset
"""
from pathlib import Path
import json
from typing import Optional 
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
import random
import cv2
import random
import cv2
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import os
from pathlib import Path

class PolypsDataset(Dataset):
    def __init__(self, split_file: Path, base_path: Path, split_type: str, transform=None, save_dir: Optional[Path] = None, colorTransfer: bool = False):
        self.base_path = base_path
        self.transform = transform
        self.data = self.load_splits(split_file, split_type)
        self.split_type = split_type
        self.save_dir = save_dir
        self.colorTransfer = colorTransfer

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def load_splits(self, split_file, split_type):
        with open(split_file, 'r') as f:
            splits = json.load(f)
        return splits[split_type]

    def get_mean_and_std(self, img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    def color_transfer(self, source, target):
        source = np.array(source)
        target = np.array(target)

        height = min(source.shape[0], target.shape[0])
        source_resized = cv2.resize(source, (int(source.shape[1] * height / source.shape[0]), height))
        target_resized = cv2.resize(target, (int(target.shape[1] * height / target.shape[0]), height))

        # RGB to LAB
        lab1 = cv2.cvtColor(source_resized, cv2.COLOR_RGB2LAB)
        lab2 = cv2.cvtColor(target_resized, cv2.COLOR_RGB2LAB)

        mean1, std1 = self.get_mean_and_std(lab1)
        mean2, std2 = self.get_mean_and_std(lab2)

        # Adjust LAB values
        adjusted_lab2 = (lab2 - mean2) / std2 * std1 + mean1

        np.putmask(adjusted_lab2, adjusted_lab2 > 255, 255)
        np.putmask(adjusted_lab2, adjusted_lab2 < 0, 0)

        # Convert LAB to RGB
        new_image2 = cv2.cvtColor(cv2.convertScaleAbs(adjusted_lab2), cv2.COLOR_LAB2RGB)

        return new_image2
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx] 

        image_fname, centre, mask_fname, chromo = (
            sample["image"],
            sample["centre"],
            sample["mask"],
            sample["chromo"],
        )

        image_path = (
            self.base_path / f"data_{centre}" / f"images_{centre}" / image_fname
        ).with_suffix(".jpg")

        mask_path = (
            self.base_path / f"data_{centre}" / f"masks_{centre}" / mask_fname
        ).with_suffix(".jpg")

        image = io.imread(image_path)  # numpyarrays[C,H,W]::uint8 \in [0, 255].
        label = io.imread(mask_path)  # Tensor[H,W]::uint8 \in [0, 255].

        label = (np.where(label > 128, 1, 0)).astype(np.uint64)
        
        # Apply color transfer during training only
        if self.split_type == 'train':
            if self.colorTransfer:

                seed_value = 42 
                random.seed(seed_value + idx) 
                ref_idx = random.randint(0, len(self.data) - 1)

                while ref_idx == idx:
                    ref_idx = random.randint(0, len(self.data) - 1)
                ref_sample = self.data[ref_idx]
                ref_image_fname, ref_centre = ref_sample["image"], ref_sample["centre"]
                if ref_image_fname.endswith("_fake"):
                    suffix = ".png"
                else:
                    suffix = ".jpg"
                ref_image_path = (self.base_path / f"data_{ref_centre}" / f"images_{ref_centre}" / ref_image_fname).with_suffix(suffix)
                ref_image = io.imread(ref_image_path)
                image = self.color_transfer(ref_image, image)

        if self.transform:
            data = self.transform(image=image, mask=label)
            data["mask"] = data["mask"].type(torch.LongTensor)
            return (data["image"], data["mask"], centre, chromo)
        else:
            return (image, label, centre, chromo)

class PolypgenDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        split_file: Path,  
        train_batch_size: int,
        num_workers: int,
        seed: int,
        base_path: Path,
        save_dir: Path = None,
        colorTransfer: bool = False,
    ) -> None:
        super().__init__()
        self.split_file = split_file
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.base_path = base_path
        self.save_dir = save_dir
        self.colorTransfer = colorTransfer

    def get_train_transforms(self) -> A.Compose:        
        return A.Compose([
            A.Resize(224, 224), 
            A.Normalize(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3), 
            ToTensorV2(),
        ])

    def get_val_transforms(self) -> A.Compose:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ])

    def setup(self) -> None:
        """Function to get pytorch datasets for train, validation and in/out testing"""

        self.train_transforms = self.get_train_transforms() 
        self.val_transforms = self.get_val_transforms()

        self.train_dataset = PolypsDataset(
            split_file=self.split_file,
            base_path=self.base_path,
            split_type='train',
            transform=self.train_transforms,
            save_dir=self.save_dir / 'train' if self.save_dir else None,
            colorTransfer=self.colorTransfer,
        )
        self.val_dataset = PolypsDataset(
            split_file=self.split_file,
            base_path=self.base_path,
            split_type='val',
            transform=self.val_transforms,
            save_dir=self.save_dir / 'val' if self.save_dir else None,
        )
        self.test_dataset = PolypsDataset(
            split_file=self.split_file,
            base_path=self.base_path,
            split_type='test',
            transform=self.val_transforms,
            save_dir=self.save_dir / 'test' if self.save_dir else None,
        )

        # Debugging: Print dataset sizes after filtering
        print(f"Train dataset: {len(self.train_dataset)}")
        print(f"Validation dataset: {len(self.val_dataset)}")
        print(f"Test dataset: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Function to get train dataloader"""
        return DataLoader(
            self.train_dataset,
            self.train_batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True, 
        )

    def val_dataloader(self) -> DataLoader: 
        "Function to get validation dataloader"
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Function to get in-distribution test dataloader"""
        dist_dl = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dist_dl

    