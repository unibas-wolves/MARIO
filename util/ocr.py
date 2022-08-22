import pytesseract
import cv2

import json
import os

DATASET_PATH = "../dataset/GORE_2022/"

pushing_frames_per_video = {}
for f in os.listdir(DATASET_PATH):
    video_path = os.path.join(DATASET_PATH, f)

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    pushing_frames_per_video[video_path] = {
        "frame_count"   : frame_count,
        "pushing_frames": []
    }

    current_frame = 0
    while True:
        return_value, frame = video.read()
        if not return_value: break
        # Check if either the word "pushing" or the word "foul" is present 
        # in the current frame. This operation is performed each ~10 seconds.
        if not current_frame % (fps * 10):
            print(f"Processing frame {current_frame} of {frame_count}...")

            detections = pytesseract.image_to_data(
                frame, 
                output_type=pytesseract.Output.DICT
            )["text"]
            if "pushing" in detections or "foul" in detections: 
                pushing_frames_per_video[video_path]["pushing_frames"].append(
                    current_frame
                )

        current_frame += 1

with open("./ocr_result.json", "w") as f:
    f.write(json.dumps(pushing_frames_per_video))
