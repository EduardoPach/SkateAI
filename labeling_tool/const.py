import os 
from pathlib import Path

import utils

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
    "video_path", 
    "video_title", 
    "video_url",
    "video_source", 
    "trick_interval", 
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

DATA_DIR_PATH = Path("data")
METADATA_DIR = Path("data/metadata")
VIDEOS_DIR = Path("data/videos")
TRICKS_JSON_PATH = Path("data/tricks_cut.json")
TRICK_NAMES_PATH = Path("data/TRICK_NAMES.json")
METADATA_FILE = Path("data/metadata/metadata.csv")
VIDEO_SOURCES_PATH = Path("labeling_tool/videos_sources.json")

TRICK_DATA = utils.get_cuts_data()
TRICK_NAMES = utils.load_json(TRICK_NAMES_PATH)
VIDEO_SOURCES = utils.load_json(VIDEO_SOURCES_PATH) 