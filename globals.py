import os 

import utils

VIDEO_PATH = utils.get_videos_url(["https://www.youtube.com/playlist?list=PLmxvVi4Ors7aqc726ngHq1SwTrBGPjCSN"]) 
STANCES = ["regular", "switch", "fakie", "nollie"]
ROTATION_TYPE = ["none", "backside", "frontside"]
FLIP_TYPE = ["none", "kickflip", "heelflip"]
NUMBER_ROTATION = ["none", "once", "twice", "thrice"]
TRICK_DATA = utils.get_cuts_data()
TRICK_NAMES = utils.load_json("data/TRICK_NAMES.json")