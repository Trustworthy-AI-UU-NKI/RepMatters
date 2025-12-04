# UM-Net training strategy implemented from paper description: https://www.sciencedirect.com/science/article/pii/S136184152400272X
# --------------------------------------------------------------------------------------------------------------------------------
from typing import Any, Optional
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import wandb
from pathlib import Path
from omegaconf import DictConfig
from utils.model_utils.losses import dice_loss
import logging
from torch.nn import functional as F
from torch.nn.functional import conv2d

class LightningTraining(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        img_logger: pl.Callback,
        save_path: Path,
        args: DictConfig = None,
    ):
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
        out1, out2, out3, out4, out5, p_c = self.model(imgs)  

        # 1. reshape decoder outputs to match labels
        outs = []
        for out in [out1, out2, out3, out4, out5]:
            if out.shape != labels.shape:
                if out.shape[1] > 1:
                    out = F.interpolate(out[:,1,:,:].unsqueeze(1), size=[224,224], mode='bilinear', align_corners=True)
                else:
                    out = F.interpolate(out, size=[224,224], mode='bilinear', align_corners=True)
            outs.append(out)
        
        # 2. calculate average prediction --> Y_avg
        preds = torch.mean(torch.stack(outs), dim=0) 

        # 3. calculate predictive uncertainty (entropy) --> U
        entropy = -torch.sum(torch.sigmoid(preds) * torch.log(torch.sigmoid(preds) + 1e-8), dim=1, keepdim=True)

        # 4. calculate discrepancy --> V_ari
        discrepancies = []
        for i in range(len(outs)):
            out = torch.sigmoid(outs[i].squeeze(1))
            discrepancy = torch.log(1 + (out - labels) ** 2)
            discrepancies.append(discrepancy)

        # 5. calculate w
        ws = torch.exp(-1 * torch.stack(discrepancies, dim=0))

        # 6. calculate 0.5 * dice loss + 0.5 * bce loss = segmentation loss
        losses_segmentation = []
        for i in range(len(outs)):
            out = outs[i]
            out_sigmoid = torch.sigmoid(out)
            loss_dice = dice_loss(out_sigmoid, labels)
            loss_bce = F.binary_cross_entropy_with_logits(out.squeeze(1), labels.float())
            loss = 0.5 * loss_dice + 0.5 * loss_bce
            losses_segmentation.append(loss)

        # 7. calculate variance loss
        variance_loss = torch.mean(torch.stack(discrepancies, dim=0))

        # 8. calculate focal loss (boundary refinement)
        sobel_x = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=labels.dtype, device=labels.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=labels.dtype, device=labels.device).view(1, 1, 3, 3)

        if labels.shape[1] > 1:
            mask = labels[:, 1:2, ...]
        else:
            mask = labels

        mask = mask.float()

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.shape[1] != 1:
            mask = mask[:, :1, ...]

        sobel_x = sobel_x.to(dtype=mask.dtype)
        sobel_y = sobel_y.to(dtype=mask.dtype)
        grad_x = conv2d(mask, sobel_x, padding=1)
        grad_y = conv2d(mask, sobel_y, padding=1)
        boundary_mask = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        boundary_mask = (boundary_mask > 0).float()

        if p_c.shape != boundary_mask.shape:
            p_c_resized = F.interpolate(p_c, size=boundary_mask.shape[2:], mode='bilinear', align_corners=True)
        else:
            p_c_resized = p_c
        focal_loss = F.binary_cross_entropy_with_logits(p_c_resized, boundary_mask)

        # 9. calculate total loss
        segmentation_final_loss = 0
        for i in range(len(losses_segmentation)):
            segm_loss = losses_segmentation[i]
            segm_loss = ws[i] * segm_loss
            segmentation_final_loss += segm_loss
        
        # segmentation_final_loss to a scalar 
        segmentation_final_loss = segmentation_final_loss.mean()
        loss = segmentation_final_loss + variance_loss + focal_loss

        dice_score = torchmetrics.functional.dice(
            out1, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        self.log("train_dice", dice_score, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        imgs, labels, _, _ = batch
        out1, out2, out3, out4, out5, p_c = self.model(imgs)  

        # 1. reshape outputs to match labels
        outs = []
        for out in [out1, out2, out3, out4, out5]:
            if out.shape != labels.shape:
                if out.shape[1] > 1:
                    out = F.interpolate(out[:,1,:,:].unsqueeze(1), size=[224,224], mode='bilinear', align_corners=True)
                else:
                    out = F.interpolate(out, size=[224,224], mode='bilinear', align_corners=True)
            outs.append(out)
        
        # 2. calculate average prediction
        preds = torch.mean(torch.stack(outs), dim=0)

        # 3. calculate predictive uncertainty (entropy)
        entropy = -torch.sum(torch.sigmoid(preds) * torch.log(torch.sigmoid(preds) + 1e-8), dim=1, keepdim=True)

        # 4. calculate discrepancy
        discrepancies = []
        for i in range(len(outs)):
            out = torch.sigmoid(outs[i].squeeze(1)) 
            discrepancy = torch.log(1 + (out - labels) ** 2)
            discrepancies.append(discrepancy)

        # 5. calculate w
        ws = torch.exp(-1 * torch.stack(discrepancies, dim=0))

        # 6. calculate 0.5*dice loss + 0.5*bce loss = segmentation loss
        losses_segmentation = []
        for i in range(len(outs)):
            out = outs[i]
            out_sigmoid = torch.sigmoid(out)
            loss_dice = dice_loss(out_sigmoid, labels)
            loss_bce = F.binary_cross_entropy_with_logits(out.squeeze(1), labels.float())
            loss = 0.5 * loss_dice + 0.5 * loss_bce
            losses_segmentation.append(loss)

        # 7. calculate variance loss
        variance_loss = torch.mean(torch.stack(discrepancies, dim=0))

        # 8. calculate focal loss (boundary refinement)
        sobel_x = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=labels.dtype, device=labels.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=labels.dtype, device=labels.device).view(1, 1, 3, 3)

        if labels.shape[1] > 1:
            mask = labels[:, 1:2, ...]
        else:
            mask = labels
        mask = mask.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.shape[1] != 1:
            mask = mask[:, :1, ...]

        sobel_x = sobel_x.to(dtype=mask.dtype)
        sobel_y = sobel_y.to(dtype=mask.dtype)
        grad_x = conv2d(mask, sobel_x, padding=1)
        grad_y = conv2d(mask, sobel_y, padding=1)
        
        boundary_mask = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        boundary_mask = (boundary_mask > 0).float()

        if p_c.shape != boundary_mask.shape:
            p_c_resized = F.interpolate(p_c, size=boundary_mask.shape[2:], mode='bilinear', align_corners=True)
        else:
            p_c_resized = p_c
        focal_loss = F.binary_cross_entropy_with_logits(p_c_resized, boundary_mask)

        # 9. calculate total loss
        segmentation_final_loss = 0
        for i in range(len(losses_segmentation)):
            segm_loss = losses_segmentation[i]
            segm_loss = ws[i] * segm_loss
            segmentation_final_loss += segm_loss

        # segmentation_final_loss to a scalar
        segmentation_final_loss = segmentation_final_loss.mean()
        loss = segmentation_final_loss + variance_loss + focal_loss

        dice_score = torchmetrics.functional.dice(
            out1, labels, zero_division=1, num_classes=2, ignore_index=0
        )

        self.log_dict({"val_dice": dice_score, "val_loss": loss})
        return out1

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
        out1, _, _, _, _, _ = self.model(imgs)  

        dice_score = torchmetrics.functional.dice(
            out1, labels, zero_division=1, num_classes=2, ignore_index=0
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(out1, 1),
            labels,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(out1, 1),
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
        return out1
 
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
