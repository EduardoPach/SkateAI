import os 

VIDEO_PATH = [os.path.join("videos", i) for i in os.listdir("videos")]
STANCES = ["regular", "switch", "fakie", "nollie"]
BODY_ROTATION_TYPE = ["frontside", "backside", "none"]