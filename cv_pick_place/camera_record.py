from robot_cell.detection.realsense_depth import DepthCamera
import numpy as np
import cv2
import time
from datetime import datetime
import os


camera = DepthCamera(config_path="config\D435_camera_config.json")
fps = 10
recording = None
just_started = True
record = False
folder = "recordings"

os.makedirs(folder, exist_ok=True)

while True:
    success, depth_frame, rgb_frame, colorized_depth = camera.get_frames()
    if not success:
        continue

    image_frame = rgb_frame.copy()
    frame_height, frame_width, frame_channel_count = rgb_frame.shape
    image_frame = cv2.resize(image_frame, (frame_width, frame_height))
    cv2.imshow("RGB Frame", image_frame)

    rgb_frame = np.expand_dims(rgb_frame, axis=3)
    depth_frame = np.expand_dims(depth_frame, axis=2)
    depth_frame = np.expand_dims(depth_frame, axis=3)
    new_frame = np.concatenate((rgb_frame, depth_frame), axis=2)
    if record:
        if just_started:
            recording = new_frame
            just_started = False
        else:
            recording = np.concatenate((recording, new_frame), axis=3)

    time.sleep(1 / fps)
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    key = cv2.waitKey(1)
    if key == 27:  # 'Esc'
        camera.release()
        print("[INFO]: Finished")
        break

    if key == ord("b"):
        print(f"[INFO]: Begin recording")
        record = True
    if key == ord("e"):
        print(f"[INFO]: End recording")
        record = False
    if key == ord("r"):
        print(f"[INFO]: Reset recording")
        just_started = True
    if key == ord("s"):
        recording_name = f"recording_{fps}fps_{date_time}.npy"
        save_file = os.path.join(folder, recording_name)
        print(f"[INFO]: Saving to file - {recording_name}")

        with open(save_file, "wb") as f:
            np.save(f, recording)
    if key == ord("h"):
        print(
            f"[INFO]: Control keys\n\tb - begin recording\n\te - end recording\n\tr - reset recorfing\n\ts - save recording"
        )
