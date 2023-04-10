from __future__ import annotations

import os
import json
import string
import shutil
from typing import Any
from pathlib import Path

import boto3
import wandb
import pytube as yt
import pandas as pd
import moviepy.editor as mo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.labeling_tool import const

load_dotenv()


def check_file_exists_in_s3(filename: str) -> bool:
    """Check if a file exists in S3.

    Parameters
    ----------
    filename : str
        The name of the file to check.

    Returns
    -------
    bool
        Whether or not the file exists in S3.
    """
    # Create an S3 client.
    s3 = boto3.client('s3')

    # Attempt to get the object's metadata.
    try:
        s3.head_object(Bucket=os.environ["S3_BUCKET"], Key=filename)
        return True
    except:
        return False

def upload_df_to_s3(df: pd.DataFrame, filepath: str) -> None:
    """Upload a DataFrame to S3 as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to upload to S3
    filepath : str
        Path of the file on S3
    """
    # Convert the DataFrame to a CSV string.
    csv_string = df.to_csv(index=False)

    # Create an S3 client.
    s3 = boto3.client('s3')

    # Upload the CSV string to S3.
    s3.put_object(Body=csv_string, Bucket=os.environ["S3_BUCKET"], Key=filepath)

def upload_mp4_to_s3(filepath: str, directory: str=".", remove_local: bool=True, **kwargs) -> None:
    """Upload mp4 file to S3 and remove it from local machine.

    Parameters
    ----------
    filepath : str
        filepath on local machine to upload to S3
    directory : str
        directory to store the file on S3, by default "."
    remove_local : bool
        whether or not to remove the file from local machine, by default True
    """
    BUCKET_NAME = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")

    # Get the filename from the filepath
    filename = os.path.basename(filepath)

    # Upload the file to S3
    try:
        s3.upload_file(filepath, BUCKET_NAME, f"{directory}/{filename}", ExtraArgs={"Metadata": kwargs})
        print(f"Successfully uploaded {filename} to S3 bucket {BUCKET_NAME}\n")
    except Exception as e:
        print(f"Error uploading {filename} to S3 bucket {BUCKET_NAME}: {e}\n")
    if remove_local:
        os.remove(filepath)

def download_video(video_title: str, video_url: str, source: str, directory: str) -> dict[str, str]:
    """Download raw video from Youtube and store it locally on directory.

    Parameters
    ----------
    video_title : str
        Video titel
    video_url : str
        Video url
    source : str
        Video source i.e. it's playlist name
    directory : str, optional
        Directory to store the video locally

    Returns
    -------
    dict[str, str]
        Video information
    """
    video = yt.YouTube(video_url)
    video_length = video.length
    video_title_parsed = parse_video_title(video_title)

    if not os.path.exists(directory):
        os.mkdir(directory)
    filepath = f"./{video_title_parsed}.mp4"


    video.streams.filter(
        res="480p",
        file_extension='mp4',
        type="video",
        only_video=True
    )[0].download(output_path=directory, filename=filepath)

    return {
        "filepath": f"{directory}/{filepath}",
        "video_url": video_url,
        "video_source": source,
        "video_title": video_title,
        "video_title_parsed": video_title_parsed,
        "video_length": str(video_length)
    }


def get_videos_url(url: str) -> dict:
    """Get's all URLs for each video in the playlist as well
    with the video's title.

    Parameters
    ----------
    urls : str
        A list containing URLs of playlists

    Returns
    -------
    dict[str, str]
        A dictionary with the title of the videos as keys 
        and their URLs as values.
    """
    data = {}
    playlist = yt.Playlist(url)
    for title, video_url in zip([video.title for video in playlist.videos],playlist.video_urls):
        data[title] = video_url
    return data

def parse_video_title(title: str) -> str:
    """Parses the video title so every video has a standard 
    title structure.

    Parameters
    ----------
    title : str
        The video title

    Returns
    -------
    str
        Parsed video title
    """
    paresed_title = ""
    for s in title:
        if s not in string.punctuation:
            paresed_title+=s
    return paresed_title.replace(" ", "_")

def initialize_data_dir(download_all: bool) -> None:
    """Initializes all the directories needed if they don't exist

    Parameters
    ----------
    download_all : bool
        Whether or not to download all videos. Forces a new metadata
        csv file to be create.
    """
    if not os.path.exists(const.DATA_DIR_PATH):
        os.mkdir(const.DATA_DIR_PATH)
    if not os.path.exists(const.VIDEOS_DIR):
        os.mkdir(const.VIDEOS_DIR)
    if not os.path.exists(const.METADATA_DIR):
        os.mkdir(const.METADATA_DIR)
    if download_all or not os.path.exists(const.METADATA_FILE):
        df = pd.DataFrame(columns=const.METADATA_COLS)
        df.to_csv(const.METADATA_FILE, index=False)

def update_metadata(video_file: str, video_title: str, video_url: str, cut_info: dict) -> None:
    """Updates/Create metadata about the cuts that were generated.

    Parameters
    ----------
    video_file : str
        The name of the mp4 file to the video cut
    video_title : str
        The title of the video from where the cut was generated
    video_url : str
        The URL of the video
    cut_info : dict
        The info about the specific cut with format
            {
                "interval": [float, float],
                "video_source": str,
                "trick_info": dict[str, Any]
            }
    """
    df = pd.read_csv("data/metadata/metadata.csv")
    entry = {
        "video_file": video_file,
        "video_title": video_title,
        "video_url": video_url,
        "video_source": cut_info["video_source"],
        "clip_start": cut_info["interval"][0],
        "clip_end": cut_info["interval"][-1],
    }
    for key, value in cut_info["trick_info"].items():
        entry[key] = value
    df = df.append(entry, ignore_index=True).reset_index(drop=True)
    df.to_csv(const.METADATA_FILE, index=False)

def split_videos(stratify_on: list=["landed", "stance"], train_size: float=0.8, remove_single: bool=True) -> None:
    """_summary_

    Parameters
    ----------
    stratify_on : list, optional
        _description_, by default ["landed", "stance"]
    train_size : float, optional
        _description_, by default 0.8
    remove_single : bool, optional
        _description_, by default True
    """
    df = pd.read_csv(const.METADATA_FILE)
    df = df.groupby(stratify_on).filter(lambda x: x.size() > 1)
    train_df, val_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[stratify_on]
    )
    train_df.to_csv(const.METADATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(const.METADATA_DIR / "validation_split.csv", index=False)

def categorical_encoder(label: str, value: str) -> int:
    """Encodes the categorical target values.

    Parameters
    ----------
    label : str
        Name of the target variable to be encoded
    value : str
        The categorical value for that specific variable

    Returns
    -------
    int
        Encoded representation of the categorical value
    """
    return const.CATEGORICAL_ENCODER[label][value]

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        val = json.load(f)
    return val

def get_cut_name(data: dict, video_url: str, trick_name: str, landed: str, stance: str) -> str:
    """Generates a standard name for the cut based on its atributes

    Parameters
    ----------
    data : dict
        The current state of the general JSON file
    video_url : str
        The URL of the video being labeled
    trick_name : str
        The name of the trick
    landed : str
        Wheter or not the trick was landed
    stance : str
        The stance of the trick

    Returns
    -------
    str
        The name that will be used for the cut
    """
    new_cut_base_name = f"{stance} {trick_name} {'landed' if landed else 'not landed'}"
    cuts_in_video = data.get(video_url, []).copy()
    counter = 0
    for cut in cuts_in_video:
        if new_cut_base_name in cut:
            counter+=1
    
    return f"{new_cut_base_name} {counter+1}"


def download_data_pipeline(download_all: bool, split: bool, stratify_on: list, train_size: float):
    """_summary_

    Parameters
    ----------
    download_all : bool
        _description_
    split : bool
        _description_
    stratify_on : list
        _description_
    train_size : float
        _description_

    Raises
    ------
    e
        _description_
    """
    #TODO use S3 or local source to create cuts
    #TODO Maybe rename this function because it deals only with labeled data
    initialize_data_dir(download_all)
    # TRICK_CUTS format {"url": "cut_name": {"interval": [start, end], "trick_info": {...}}}
    n_cuts = sum(len(video_cuts) for video_cuts in const.TRICK_CUTS.values())
    n_downloaded_videos = len(os.listdir(const.VIDEOS_DIR))
    print(f"THERE ARE {n_cuts - n_downloaded_videos} NEW CUTS TO BE DOWNLOADED\n")
    total = 1
    
    for url, cuts in const.TRICK_CUTS.items():
        counter = 1
        tries = 0
        while tries < 10: # Workaround pytube issue
            try:
                video = yt.YouTube(url)
                video_title = video.title
                video_title = parse_video_title(video_title)
                break
            except Exception as e:
                tries +=1
                if tries==10:
                    raise e
        print("#"*100)
        print(f"DOING {video_title.upper()}\n")
        for cut_name, cut_info in cuts.items():
            video_file = f"{counter:05}.mp4"
            clips_dir = const.VIDEOS_DIR / video_title
            video_path = clips_dir / video_file
            fullvideo = "fullvideo.mp4"
            fullvideo_path = const.VIDEOS_DIR / fullvideo
            if not fullvideo_path.exists():
                print("DOWNLOADING FULL VIDEO:", end=" ")
                video.streams.filter(
                    res="480p",
                    file_extension='mp4',
                    type="video",
                    only_video=True
                )[0].download(output_path=const.VIDEOS_DIR, filename=fullvideo)
                print("Success!\n")
            print("-"*100)
            print(f"DOWNLOADING {cut_name} AS {video_file} - VIDEO {total} OUT OF {n_cuts} ({100*total/n_cuts:.1f}%)\n")
            os.makedirs(clips_dir, exist_ok=True)
            if os.path.exists(video_path) and not download_all:
                print("\nALREADY EXISTS!\n")
                print("-"*100)
                counter += 1
                total += 1
                continue

            print("CUTTING VIDEO: ", end='')
            with mo.VideoFileClip(str(const.VIDEOS_DIR/fullvideo)) as f:
                clip = f.subclip(*cut_info["interval"])
                clip.write_videofile(str(video_path))
            
            update_metadata(video_file, video_title, url, cut_info)
            print("Success!")
        
            counter += 1
            total += 1

        os.remove(fullvideo_path)

    if not split: 
        return
    print("#"*100)
    print(f"SPLITTING DATASET IN TRAIN AND TEST ({train_size*100:.2f} / {(1-train_size)*100:.2f}) \
        W/ STRATIFICATION ON {', '.join(stratify_on)}")
    split_videos(stratify_on, train_size)

def create_table_with_videos(df: pd.DataFrame) -> wandb.Table:
    columns = df.columns.tolist()
    columns = ["video"] + columns
    table = wandb.Table(columns=columns)
    data = df.to_numpy().tolist()
    for row in data:
        video_file = row[0]
        video_dir = row[1]
        video = wandb.Video(f"{const.VIDEOS_DIR}/{video_dir}/{video_file}", format="mp4")
        table.add_data(video, *row)
    return table

def wandb_log_dataset() -> None:
    with wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], job_type="upload") as run:
        raw_data = wandb.Artifact(os.environ["WANDB_DATASET_ARTIFACT"], type="dataset")
        raw_data.add_dir(const.VIDEOS_DIR, name="videos")
        metadata = pd.read_csv(const.METADATA_FILE)
        table = create_table_with_videos(metadata)
        raw_data.add(table, "eda_table")
        run.log_artifact(raw_data)

def wandb_log_split() -> None:
    with wandb.init(project=os.environ["WANDB_PROJECT"], job_type="data_split") as run:
        split = wandb.Artifact(os.environ["WANDB_SPLIT_ARTIFACT"], type="split_data")

        train_df = pd.read_csv(const.METADATA_DIR / "train_split.csv")
        train_df["split"] = "train"
        train_df = train_df[["video_file", "split"]]
        val_df = pd.read_csv(const.METADATA_DIR / "validation_split.csv")
        val_df["split"] = "validation"
        val_df[["video_file", "split"]]

        df = pd.concat([train_df, val_df]).reset_index(drop=True)
        split_table = wandb.Table(dataframe=df)

        raw_data = run.use_artifact(f"{os.environ['WANDB_DATASET_ARTIFACT']}:latest")
        eda_table = raw_data.get("eda_table")
        path = Path(raw_data.download())

        join_table = wandb.JoinedTable(eda_table, split_table, "video_file")
        split.add(join_table, "eda_table_data_split")
        split.add_dir(const.METADATA_DIR, name="metadata")

        shutil.rmtree(path.parent)
        run.log_artifact(split)