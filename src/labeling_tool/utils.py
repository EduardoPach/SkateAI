import os
import json
from typing import Any

from src.labeling_tool import const

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

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        val = json.load(f)
    return val