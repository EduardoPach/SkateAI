import os 

VIDEO_PATH = [os.path.join("videos", i) for i in os.listdir("videos")]
STANCES = ["regular", "switch", "fakie", "nollie"]
ROTATION_TYPE = ["none", "backside", "frontside"]
FLIP_TYPE = ["none", "kickflip", "heelflip"]
NUMBER_ROTATION = ["none", "once", "twice", "thrice"]