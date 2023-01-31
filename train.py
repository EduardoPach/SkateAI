from __future__ import annotations

import os
import warnings
warnings.simplefilter("ignore", UserWarning)

import yaml
import wandb
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet
import torchvision.transforms as transforms
from sklearn.preprocessing import OrdinalEncoder

from utils import train_fn, get_loaders, load_checkpoint, save_checkpoint, check_performance, wandb_log_model
from models import ResNet18_RNN


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"],config=config, job_type="train") as run:
        raw_data_artifact = run.use_artifact(os.environ["WANDB_DATASET_ARTIFACT"]+":latest")
        split_artifact = run.use_artifact(os.environ["WANDB_SPLIT_ARTIFACT"]+":latest")
        raw_data_artifact.download("./data")
        split_artifact.download("./data")
        config = wandb.config

        EPOCHS = config["training_parameters"]["epochs"]
        LEARNING_RATE = config["training_parameters"]["learning_rate"]
        MODEL_SAVE_PATH = config["training_parameters"]["model_save_path"]
        LABEL_COLUMNS = config["training_parameters"]["label_columns"]

        TRAIN_CSV = config["dataloader_parameters"]["train_csv"]
        VAL_CSV = config["dataloader_parameters"]["val_csv"]
        ROOT_DIR = config["dataloader_parameters"]["root_dir"]
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        BATCH_SIZE = config["dataloader_parameters"]["batch_size"]
        MAX_FRAMES = config["dataloader_parameters"]["max_frames"]
        NUM_WORKERS = config["dataloader_parameters"]["num_workers"]
        PIN_MEMORY = config["dataloader_parameters"]["pin_memory"]

        RNN_TYPE = config["model_parameters"]["rnn_type"]
        RNN_LAYERS = config["model_parameters"]["rnn_layers"]
        RNN_HIDDEN = config["model_parameters"]["rnn_hidden"]
        TRAINABLE_BACKBONE = config["model_parameters"]["trainable_backbone"]
        HEADS_PARAMS = config["model_parameters"]["heads_params"]
        HEADS_PARAMS["in_features"] = RNN_HIDDEN * MAX_FRAMES

        model = ResNet18_RNN(
            RNN_TYPE, 
            RNN_LAYERS, 
            RNN_HIDDEN, 
            HEADS_PARAMS, 
            TRAINABLE_BACKBONE
        )

        resnet_transforms = resnet.ResNet18_Weights.DEFAULT.transforms()

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        loss_fns = {
            'trick_name': nn.CrossEntropyLoss(),
            'landed': nn.CrossEntropyLoss(),
            'stance': nn.CrossEntropyLoss()
        }

        train_transforms = transforms.Compose([
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            resnet_transforms,
        ])

        val_transforms = transforms.Compose([
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            resnet_transforms,
        ])

        train_labels = pd.read_csv(TRAIN_CSV)[LABEL_COLUMNS]
        encoder = OrdinalEncoder().set_output(transform="pandas")
        encoder.fit(train_labels)

        train_loader, val_loader = get_loaders(
            TRAIN_CSV,
            VAL_CSV,
            ROOT_DIR,
            MAX_FRAMES,
            BATCH_SIZE,
            train_transforms,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY,
            encoder
        )

        model.to(DEVICE)
        for epoch in range(EPOCHS):
            model.train()
            train_loss = train_fn(train_loader, model, optimizer, loss_fns, DEVICE)
            val_loss = check_performance(val_loader, model, loss_fns, DEVICE)
            val_loss = {f"{key}_val": val for key, val in val_loss.items()}
            train_loss = {f"loss_{key}_train": val for key, val in train_loss.items()}
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, MODEL_SAVE_PATH)
            wandb.log({**train_loss, **val_loss, "epoch": epoch+1})

        wandb_log_model(model_path=MODEL_SAVE_PATH, name="conv_lstm", type="model", description="ResNet18 as backbone")

if __name__=="__main__":
    main()