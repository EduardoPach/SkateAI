import pandas as pd
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

def detect_and_draw_box(img, model="yolov3-tiny", confidence=0.5):
    bbox, label, conf = cv.detect_common_objects(img, confidence=confidence, model=model)
    
    # Print detected objects with confidence level
    for l, c in zip(label, conf):
        print(f"Detected object: {l} with confidence level of {c}\n")
    # Create a new image that includes the bounding boxes
    output_image = draw_bbox(img, bbox, label, conf)
    return output_image
    

df = pd.read_csv("data/metadata/metadata.csv")
video = paths = df["video_path"].values[0]


cap = cv2.VideoCapture(video)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    output_image = detect_and_draw_box(frame, confidence=0.75)

    cv2.imshow("frame", output_image)

    key = cv2.waitKey(1)