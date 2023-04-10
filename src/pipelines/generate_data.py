from __future__ import annotations

import argparse

from src.pipelines import utils


parser = argparse.ArgumentParser(description="Command line downloading the data")

parser.add_argument(
    "--download-all",
    action="store_true",
    help="Wheter or not to download all videos"
)

parser.add_argument(
    "--split",
    action="store_true",
    help="Wheter or not to split dataset after download",
    default=False
)

parser.add_argument(
    "--stratify-on",
    nargs="+",
    type=str,
    help="Which columns use when stratifying split",
    default=["stance", "landed"]
)

parser.add_argument(
    "--train-size",
    type=float,
    help="The train size as fraction of the total dataset i.e. between 0 and 1",
    default=0.8
)

parser.add_argument(
    "--wandb-log",
    action="store_true",
    help="Wheter or not to versioning data with Weights & Biases",
    default=False
)

def main(download_all: bool, split: bool, stratify_on: list, train_size: float, wandb_log: bool) -> None:
    utils.download_data_pipeline(download_all, split, stratify_on, train_size)
    if wandb_log:
        utils.wandb_log_dataset()
        utils.wandb_log_split()

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        download_all=args.download_all,
        split=args.split,
        stratify_on=args.stratify_on,
        train_size=args.train_size,
        wandb_log=args.wandb_log
    )