import string
import os
import json

import pytube as yt
import pandas as pd

def get_videos_url(urls: list) -> dict[str, str]:
    """Get's all URLs for each video in the playlists as well
    with the video's title.

    Parameters
    ----------
    urls : list
        A list containing URLs of playlists

    Returns
    -------
    dict[str, str]
        A dictionary with the title of the videos as keys 
        and their URLs as values.
    """
    data = {}
    for url in urls:
        playlist = yt.Playlist(url)
        for title, video_url in zip([video.title for video in playlist.videos],playlist.video_urls):
            data[title] = video_url
    return data

def update_cuts(data: dict, video_url: str, cut_name: str, start_time: int, end_time: int, trick_info: dict) -> dict:
    if video_url not in data:
        data[video_url] = {
            cut_name: {
                "interval": [start_time, end_time],
                "trick_info": trick_info
            }
        }
    else:
        data[video_url][cut_name] = {"interval": [start_time, end_time], "trick_info": trick_info}
    return data

def delete_cuts(data: dict, video_url: str, current_cut: str) -> dict:
    del data[video_url][current_cut]
    return data

def get_cuts_data() -> dict:
    DIR = "batb11"
    JSON_PATH = f"{DIR}/tricks_cut.json"
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    if not os.path.exists(JSON_PATH):
        data = {}
    else:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
    return data

def parse_video_title(title: str) -> str:
    paresed_title = ""
    for s in title:
        if s not in string.punctuation:
            paresed_title+=s
    return paresed_title.replace(" ", "_")

def initialize_data_dir() -> None:
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/videos"):
        os.mkdir("data/videos")
    if not os.path.exists("data/metadata"):
        os.mkdir("data/metadata")
    if not os.path.exists("data/metadata/metadata.csv"):
        df = pd.DataFrame(columns=["video_path", "video_title", "video_url", "trick_interval", "trick_info"])
        df.to_csv("data/metadata/metadata.csv", index=False)

def update_metadata(video_path: str, video_title: str, video_url: str, trick_interval: list, trick_info: dict) -> None:
    df = pd.read_csv("data/metadata/metadata.csv")
    entry = {
        "video_path": video_path,
        "video_title": video_title,
        "video_url": video_url,
        "trick_interval": trick_interval,
        "trick_info": trick_info
    }
    df = df.append(entry, ignore_index=True).reset_index(drop=True)
    df.to_csv("data/metadata/metadata.csv", index=False)

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        val = json.load(f)
    return val