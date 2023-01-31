import os
import json
import string
import shutil
from typing import Any
from pathlib import Path

import wandb
import pytube as yt
import pandas as pd
import moviepy.editor as mo
from sklearn.model_selection import train_test_split

import const


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

def update_cuts(data: dict, video_url: str, start_time: int, end_time: int, trick_info: dict, source: str) -> dict:
    """Update general JSON file that contains the trick cuts for each video.

    Parameters
    ----------
    data : dict
        The actual state of the general JSON file
    video_url : str
        The URL of the current video being labeled
    start_time : int
        The start time of the cut in the video in seconds
    end_time : int
        The end time of the cut in the video in seconds
    trick_info : dict
        A dictionary with all the relavant information about the trick in the cut
    source : str
        The source of the video

    Returns
    -------
    dict
        An updated version of the general JSON file
    """
    cut_name = get_cut_name(
        data=data, 
        video_url=video_url, 
        trick_name=trick_info["trick_name"],
        landed=trick_info["landed"],
        stance=trick_info["stance"]
    )
    if video_url not in data:
        data[video_url] = {
            cut_name: {
                "interval": [start_time, end_time],
                "video_source": source,
                "trick_info": trick_info,
            }
        }
    else:
        data[video_url][cut_name] = {
            "interval": [start_time, end_time], 
            "video_source": source, 
            "trick_info": trick_info
        }
    return data

def delete_cuts(data: dict, video_url: str, current_cut: str) -> dict:
    """Removes an existing cut from the general JSON file

    Parameters
    ----------
    data : dict
        The current state of the JSON file
    video_url : str
        The URL of the video being labeled
    current_cut : str
        The name of the cut that will be removed

    Returns
    -------
    dict
        An updated version of the JSON file without the cut removed
    """
    del data[video_url][current_cut]
    return data

def get_cuts_data() -> dict:
    """Loads the current state of the general JSON file

    Returns
    -------
    dict
        Current state of JSON file
    """ 
    if not os.path.exists(const.DATA_DIR_PATH):
        os.mkdir(const.DATA_DIR_PATH)

    if not os.path.exists(const.TRICKS_JSON_PATH):
        data = {}
    else:
        data = load_json(const.TRICKS_JSON_PATH)
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
        "trick_interval": cut_info["interval"],
    }
    for key, value in cut_info["trick_info"].items():
        entry[key] = value
    df = df.append(entry, ignore_index=True).reset_index(drop=True)
    df.to_csv(const.METADATA_FILE, index=False)

def split_videos(stratify_on: list=["landed", "stance"], train_size: float=0.8) -> None:
    """_summary_

    Parameters
    ----------
    stratify_on : list, optional
        _description_, by default ["landed", "stance"]
    train_size : float, optional
        _description_, by default 0.8
    """
    df = pd.read_csv(const.METADATA_FILE)
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

def key_from_value(d: dict, value: Any) -> str:
    """Returns the key of a dictionary given a value. 
    Assumes that the key-value pair exists.

    Parameters
    ----------
    d : dict
        A dictionary

    value : Any
        The value associated with a key

    Returns
    -------
    str
        Key that maps to the passed value
    """
    return list(d.keys())[list(d.values()).index(value)]


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
    initialize_data_dir(download_all)
    TRICK_CUTS = get_cuts_data() # {"url": "cut_name": {"interval": [start, end], "trick_info": {...}}}
    N_CUTS = sum(len(video_cuts) for video_cuts in TRICK_CUTS.values())
    N_DOWNLOADED_VIDEOS = len(os.listdir(const.VIDEOS_DIR))
    print(f"THERE ARE {N_CUTS - N_DOWNLOADED_VIDEOS} NEW CUTS TO BE DOWNLOADED\n")
    
    for url, cuts in TRICK_CUTS.items():
        counter = 1
        video = yt.YouTube(url)
        video_title = video.title
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
            print(f"DOWNLOADING {cut_name} AS {video_file} - VIDEO {counter} OUT OF {N_CUTS} ({100*counter/N_CUTS:.1f}%)\n")
            os.makedirs(clips_dir, exist_ok=True)
            if os.path.exists(video_path) and not download_all:
                print("\nALREADY EXISTS!\n")
                print("-"*100)
                counter += 1
                continue

            print("CUTTING VIDEO: ", end='')
            with mo.VideoFileClip(str(const.VIDEOS_DIR/fullvideo)) as f:
                clip = f.subclip(*cut_info["interval"])
                clip.write_videofile(str(video_path))
            
            update_metadata(video_file, video_title, url, cut_info)
            print("Success!")
        
            counter += 1

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
        video = wandb.Video(f"{const.VIDEOS_DIR}/{video_file}", format="mp4")
        table.add_data(video, *row)
    return table

def wandb_log_dataset() -> None:
    with wandb.init(project=os.environ["WANDB_PROJECT"], job_type="upload") as run:
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