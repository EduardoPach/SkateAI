from __future__ import annotations

import os
import argparse
from glob import glob

import pandas as pd
from tqdm import tqdm
from scenedetect import (
    open_video, ContentDetector, SceneManager, 
    StatsManager, split_video_ffmpeg, FrameTimecode
)

from src.pipelines import utils
from src.pipelines.const import LOCAL_RAW_VIDEOS_DIR, LOCAL_CLIPS_DIR, AWS_S3_CLIPS_DIR

def find_scenes(filepath: str, threshold: int=35) -> list[tuple[FrameTimecode, FrameTimecode]]:
    """Split a video into scenes

    Parameters
    ----------
    filepath : str
        Video filepath on the local machine
    threshold : int, optional
        Threshold to split video, by default 35

    Returns
    -------
    list[tuple[FrameTimecode, FrameTimecode]]
        List of scenes
    """
    video = open_video(filepath)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    return scene_manager.get_scene_list()

def scenes_to_dataframe(scenes: list[tuple[FrameTimecode, FrameTimecode]], **kwargs) -> pd.DataFrame:
    """Convert a list of scenes to a dataframe

    Parameters
    ----------
    scenes : list[tuple[FrameTimecode, FrameTimecode]]
        List of scenes

    Returns
    -------
    pd.DataFrame
        Dataframe with the scenes
    """
    df = pd.DataFrame(
        data=scenes, 
        columns=["start", "end"]
    )
    df["start"] = df["start"].apply(lambda x: x.get_seconds())
    df["end"] = df["end"].apply(lambda x: x.get_seconds())
    df["duration"] = df["end"] - df["start"]
    if kwargs:
        for key, val in kwargs.items():
            df[key] = val
    return df

def main(args: argparse.Namespace) -> None:
    """_summary_
    """
    # Getting arguments
    override = args.override
    store_s3 = args.store_s3
    # Step 1 - Get List of Raw Videos
    videos = sorted(glob(f'{LOCAL_RAW_VIDEOS_DIR}/*.mp4'))
    # Step 2 - Prepare LOCAL_CLIPS_DIR if it doesn't exist
    if not os.path.exists(LOCAL_CLIPS_DIR):
        os.mkdir(LOCAL_CLIPS_DIR)
    dfs = []
    # Step 3 - For each video, find the scenes
    for video in tqdm(videos):
        # Getting filename and stripping extension
        video_name = os.path.basename(video).split(".")[0]
        path = f"{LOCAL_CLIPS_DIR}/{video_name}"
        # Check if the directory exists
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            if not override:
                continue

            scenes = find_scenes(filepath=video)

            # Step 4 - Split the video into scenes and save them
            split_video_ffmpeg(
                input_video_path=video, 
                output_file_template=f"{path}/$SCENE_NUMBER.mp4", 
                scene_list=scenes
            )

            # Step 5 - Create a dataframe with the scenes
            df = scenes_to_dataframe(scenes, video_title_parsed=video_name)
            df["filepath"] = sorted(glob(f"{path}/*.mp4"))
            dfs.append(df)
            
    # Step 6 - Concatenate all the dataframes and store locally
    df = pd.concat(dfs)
    df.to_csv(f"{LOCAL_CLIPS_DIR}/clips.csv", index=False)
    # Step 7 - Upload to S3 if needed
    if not store_s3:
        return
    
    clips = glob(f'{LOCAL_CLIPS_DIR}/*/*.mp4')

    for clip in clips:
        parent_dir = os.path.basename(os.path.dirname(clip))
        s3_dir = f"{AWS_S3_CLIPS_DIR}/{parent_dir}"
        utils.upload_mp4_to_s3(clip, directory=s3_dir, remove_local=False)

    utils.upload_df_to_s3(df, f"{AWS_S3_CLIPS_DIR}/clips.csv")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--override', action='store_true', default=True)
    parser.add_argument('--store-s3', action='store_true')
    args = parser.parse_args()
    main(args)
    