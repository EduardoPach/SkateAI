import os 

import utils

STANCES = ["regular", "switch", "fakie", "nollie"]
ROTATION_TYPE = ["none", "backside", "frontside"]
FLIP_TYPE = ["none", "kickflip", "heelflip"]
NUMBER_ROTATION = ["none", "once", "twice", "thrice"]
DEFAULT_SOURCE = "BATB 11"
VIDEO_SOURCES_PATH = "labeling_tool/videos_sources.json"
DATA_DIR_PATH = "data"
TRICK_NAMES_PATH = "data/TRICK_NAMES.json"
TRICKS_JSON_PATH = "data/tricks_cut.json"
VIDEOS_LOCAL_PATH = "data/videos"
METADATA_DIR = "data/metadata"
METADATA_FILE = "data/metadata/metadata.csv"
METADATA_COLS = ["video_path", "video_title", "video_url", "trick_interval", "trick_name", "trick_info"]

TRICK_DATA = utils.get_cuts_data()
TRICK_NAMES = utils.load_json(TRICK_NAMES_PATH)
VIDEO_SOURCES = utils.load_json(VIDEO_SOURCES_PATH) 