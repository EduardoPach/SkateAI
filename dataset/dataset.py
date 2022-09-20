import os
from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset

import const
import utils

class TricksDataset(Dataset):
    def __init__(self, video_dir: Path, transform: list=None) -> None:
        self.video_dir = video_dir
        self.videos = os.listdir(video_dir)
        self.transform = transform
        self.metadata = pd.read_csv("data/metadata/metadata.csv")
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, index):
        video_path = os.path.join(self.video_dir, self.videos[index])
        labels = self.metadata.loc[self.metadata["video_path"]==video_path, const.LABELS].to_numpy()
        frames = utils.get_video(video_path)

        if self.transform:
            frames = self.transform(frames)

        return frames, labels