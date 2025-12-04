# main script for training a U-Net model on the PolypGen dataset
from PolypgenDatamodule import PolypgenDataModule
import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import random
import torchvision.models as models
import warnings
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from lightning_training import LightningTraining   
import os

# uncomment to use U-Net*
#from unet_umnetbase import UNet

# U-Net:
from unet import UNet

warnings.filterwarnings("ignore")

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
    save_path = args.data.save_dir
    path_to_results = args.train.path_to_results

    ### Checkpoints setup ###
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(path_to_results + save_path),
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
        save_dir=Path(path_to_results + save_path),
    )

    if args.debug:
        print("Running in debug mode. Initialization complete.")
    else: 
        model = UNet(n_channels = args.unet.n_channels)
        TrainLight = LightningTraining(
            model=model,
            lr=args.train.learning_rate,
            img_logger=wandb_logger,
            save_path=Path(path_to_results + save_path),
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
                ckpt_path=args.train.resume_checkpoint_path  # Specify checkpoint path here
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
