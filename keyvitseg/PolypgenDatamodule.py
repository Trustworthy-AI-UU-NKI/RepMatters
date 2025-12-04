# Class to define a LightningDataModule for the PolypGen2021 dataset

from pathlib import Path
import json
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import albumentations as A
import pandas as pd
from albumentations.pytorch.transforms import ToTensorV2

class PolypsDataset(Dataset): 
    """Class to define a Pytorch Dataset for the PolypGen2021 dataset"""

    def __init__(
        self,
        split_file: Path, 
        base_path: Path,
        split_type: str,
        transform=None, 
    ):
        self.base_path = base_path
        self.transform = transform
        self.data = self.load_splits(split_file, split_type)

    def load_splits(self, split_file, split_type):
        with open(split_file, 'r') as f:
            splits = json.load(f)
        return splits[split_type]
    
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

        image = io.imread(image_path) 
        label = io.imread(mask_path) 
        label = (np.where(label > 128, 1, 0)).astype(np.uint64)

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

    ) -> None:
        super().__init__()
        self.split_file = split_file
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.base_path = base_path

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
    
    def setup(self, stage: str = None) -> None:
        """Function to get pytorch datasets for train, validation and in/out testing"""
        
        self.train_transforms = self.get_train_transforms() 
        self.val_transforms = self.get_val_transforms()

        self.train_dataset = PolypsDataset(
            split_file=self.split_file,
            base_path=self.base_path,
            split_type='train',
            transform=self.train_transforms,
        )
        self.val_dataset = PolypsDataset(
            split_file=self.split_file,
            base_path=self.base_path,
            split_type='val',
            transform=self.val_transforms,
        )
        self.out_test_dataset = PolypsDataset(
            split_file=self.split_file,
            base_path=self.base_path,
            split_type='test',
            transform=self.val_transforms,
        )

        # Debugging
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Ood test dataset size: {len(self.out_test_dataset)}")


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
        """Function to get out-of-distribution test dataloader"""
        out_dist_dl = DataLoader(
            self.out_test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return out_dist_dl

    