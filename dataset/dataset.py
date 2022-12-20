import os
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

import const
import utils

class TricksDataset(Dataset):
    def __init__(self, csv_file: Path, root_dir: Path, transform: list=None) -> None:
        self.video_dir = root_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_file)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, index):
        video_path = os.path.join(self.video_dir, self.metadata.iloc[index, 0])
        labels = self.metadata.loc[self.metadata["video_file"]==video_path, const.LABELS].to_numpy().astype(int)
        frames = utils.get_video(video_path)

        if self.transform:
            frames = self.transform(frames)

        return frames, labels