from PolypgenDatamodule import PolypgenDataModule
import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import random
import warnings
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from lightning_training import LightningTraining  
import os
import torch
from model import KeyViTSeg
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path=".", config_name="config") 
def main(args: DictConfig) -> None:
    print(OmegaConf.to_yaml(args)) 
    
    seed = args.train.seed

    ### Seed everything ###
    random.seed(seed)
    pl.seed_everything(seed) 

    ### Initialize logging ###
    wandb_logger = WandbLogger(  
    name=args.wandb.wandb_name,
    project=args.wandb.project_name,
)
    ### Checkpoints setup ###
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.train.path_to_results),
        save_top_k=1,
        save_last=True,
        monitor="val_dice",
        mode="max",
    )

    ### Read the Split File ###
    split_file_path = Path(args.data.split_file)

    ### Initialize LightningDataModule ###
    polypsDataset = PolypgenDataModule(
        split_file=split_file_path,
        train_batch_size=args.train.train_batch_size,
        num_workers=args.train.num_workers,
        seed=seed,
        base_path=Path(args.data.base_path),
    )

    ### Get model ###
    model = KeyViTSeg(model_type=args.extractor.model_type, version=args.extractor.version, stride =args.extractor.stride, facet = args.extractor.facet)
  
    ### Get case ###
    if args.extractor.freeze:       
        model.extractor.requires_grad_(False)  
        logging.info("EXTRACTOR WEIGHTS: frozen (case A!) ----------------------------------------")  
    else:
        logging.info("EXTRACTOR WEIGHTS: not frozen (case B!) ------------------------------------")

    ### Initialize the LightningTraining class ###
    TrainLight = LightningTraining(
        model = model,
        lr=args.train.learning_rate,
        img_logger=wandb_logger,
        save_path=Path(args.train.path_to_results),
    )

    ### Training ###
    if args.train.resume and args.train.resume_checkpoint_path:

        if not os.path.exists(args.train.resume_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {args.train.resume_checkpoint_path} does not exist.")
        
        print(f"Resuming training from {args.train.resume_checkpoint_path}...")
        
        trainer = Trainer(
            max_epochs=args.train.max_epochs,
            num_sanity_val_steps=args.train.num_sanity_val_steps,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            precision=16 if args.train.use_16bit else 32,
        )

        trainer.fit(
            TrainLight,
            datamodule=polypsDataset,
            ckpt_path=args.train.resume_checkpoint_path 
    )
        
    else:
        print("Starting training from scratch...")
        
        trainer = Trainer(
            max_epochs=args.train.max_epochs,
            num_sanity_val_steps=args.train.num_sanity_val_steps,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            precision=16 if args.train.use_16bit else 32,
        )

        trainer.fit(
            TrainLight,
            datamodule=polypsDataset,
        )

    trainer.test(TrainLight, datamodule=polypsDataset, ckpt_path="best")
            
if __name__ == "__main__":
    main()
