# LightningModule

from typing import Any, Optional
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import wandb
from pathlib import Path
from losses import *
from omegaconf import DictConfig

class LightningTraining(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        img_logger: pl.Callback,
        save_path: Path,
        args: DictConfig = None,
    ):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        self.model = model
        self.args = args
        self.lr = lr
        self.img_logger = img_logger
        self.save_path = save_path
        self.max_dice_train = -1
        self.table = []
        self.val_table = []
        self.test_table = []
        self.columns = ["dice score", "image", "ground_truth", "prediction"]
        self.columns_test = [
            "centre",
            "dice score",
            "recall",
            "accuracy",
            "image",
            "ground_truth",
            "prediction",
            "chromo",
        ]
        self.out_dice = []


    def configure_optimizers(self):
        params_to_optimize = list(self.model.parameters())
        optimizer = optim.AdamW(params_to_optimize, lr=self.lr)
        return optimizer
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        imgs, labels, _, _ = batch
        preds = self.model(imgs) 
        
        # preds_argmax = torch.argmax(preds, 1)
        loss = dice_loss(torch.sigmoid(preds)[:, 1:, ...], labels)
        
        # compute the dice score
        dice_score = torchmetrics.functional.dice(
            preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        self.log("train_dice", dice_score, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        imgs, labels, _, _ = batch
        preds = self.model(imgs)
        loss = dice_loss(torch.sigmoid(preds)[:, 1:, ...], labels)

        # compute dice score
        dice_score = torchmetrics.functional.dice(
            preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        self.log_dict({"val_dice": dice_score, "val_loss": loss})
        return preds

    def on_validation_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        imgs, labels, centre, chromo = batch

        dice = torchmetrics.functional.dice(
            outputs, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(outputs, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(outputs, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )
        preds_plot = torch.argmax(outputs, 1)
        images = [
            [
                ctr,
                dice,
                recall_score,
                accuracy_score,
                wandb.Image(img),
                wandb.Image(lbl.float()),
                wandb.Image(pred.float()),
                chromo,
            ]
            for ctr, img, lbl, pred in zip(centre, imgs, labels, preds_plot)
        ]

        self.val_table.extend(images)

    def on_train_end(self) -> None:
        self.img_logger.log_table(
            key="validation predictions", columns=self.columns, data=self.table
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, labels, _, _ = batch
        preds = self.model(imgs)  
        
        # compute dice score
        dice_score = torchmetrics.functional.dice(
            preds, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(preds, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(preds, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )

        self.log_dict(
            {
                f"{dataloader_idx}_test_dice": dice_score,
                f"{dataloader_idx}_recall": recall_score,
                f"{dataloader_idx}_accuracy": accuracy_score,
            }
        )
        print(f"Test step called for dataloader_idx={dataloader_idx}, batch_idx={batch_idx}")
        return preds
 
    def on_test_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        imgs, labels, centre, chromo = batch

        dice = torchmetrics.functional.dice(
            outputs, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(outputs, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )

        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(outputs, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )

        if outputs is not None:
            preds_plot = torch.argmax(outputs, 1)
            images = [
                [
                    ctr,
                    dice,
                    recall_score,
                    accuracy_score,
                    wandb.Image(img),
                    wandb.Image(lbl.float()),
                    wandb.Image(pred.float()),
                    chromo,
                ]
                for ctr, img, lbl, pred in zip(centre, imgs, labels, preds_plot)
            ]

            self.test_table.extend(images)

    def on_test_end(self) -> None:
        self.img_logger.log_table(
            key="test set predictions",
            columns=self.columns_test,
            data=self.test_table,
        )

        print("Test end: Logged test results to wandb")
