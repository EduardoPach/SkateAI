import os 
import json
from pathlib import Path


def get_cuts_data() -> dict:
    """Loads the current state of the general JSON file

    Returns
    -------
    dict
        Current state of JSON file
    """ 
    if not os.path.exists(DATA_DIR_PATH):
        os.mkdir(DATA_DIR_PATH)

    if not os.path.exists(TRICKS_JSON_PATH):
        data = {}
    else:
        data = load_json(TRICKS_JSON_PATH)
    return data

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        val = json.load(f)
    return val


DEFAULT_SOURCE = "BATB 11"
STANCES = ["regular", "switch", "fakie", "nollie"]
ROTATION_TYPE = ["none", "backside", "frontside"]
FLIP_TYPE = ["none", "kickflip", "heelflip"]
CATEGORICAL_ENCODER = {
    "body_rotation_type": {"none": 0, "backside": 1, "frontside": 2},
    "board_rotation_type": {"none": 0, "backside": 1, "frontside": 2},
    "flip_type": {"none": 0, "kickflip": 1, "heelflip": 2},
    "stance": {"regular": 0, "fakie": 1, "switch": 2, "nollie": 3}
}

METADATA_COLS = [
    "video_file", 
    "video_title", 
    "video_url",
    "video_source", 
    "clip_start", 
    "clip_end", 
    "trick_name", 
    "body_rotation_type",
    "body_rotation_number",
    "board_rotation_type",
    "board_rotation_number",
    "flip_type",
    "flip_number",
    "landed",
    "stance"
]

DATA_DIR_PATH = Path("src/data")
METADATA_DIR = Path("src/data/metadata")
VIDEOS_DIR = Path("src/data/videos")
TRICKS_JSON_PATH = Path("src/data/tricks_cut.json")
TRICK_NAMES_PATH = Path("src/data/TRICK_NAMES.json")
METADATA_FILE = Path("src/data/metadata/metadata.csv")
VIDEO_SOURCES_PATH = Path("src/labeling_tool/videos_sources.json")
VIDEOS_PER_SOURCE_PATH = Path("src/labeling_tool/videos_per_source.json")

TRICK_DATA = get_cuts_data()
TRICK_NAMES = load_json(TRICK_NAMES_PATH)
VIDEO_SOURCES = load_json(VIDEO_SOURCES_PATH) 
VIDEOS_PER_SOURCE = load_json(VIDEOS_PER_SOURCE_PATH)