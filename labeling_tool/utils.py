import string
import os
import json
from typing import Any

import pytube as yt
import pandas as pd

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
        if key in const.CATEGORICAL_ENCODER:
            entry[key] = categorical_encoder(key, value)
            continue
        entry[key] = int(value)
    df = df.append(entry, ignore_index=True).reset_index(drop=True)
    df.to_csv(const.METADATA_FILE, index=False)

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


    