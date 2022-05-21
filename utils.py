import os
import json

import pytube as yt

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