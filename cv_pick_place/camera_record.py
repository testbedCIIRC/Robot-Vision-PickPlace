from robot_cell.detection.realsense_depth import DepthCamera
import numpy as np
import cv2
import time

camera = DepthCamera(config_path = 'D435_camera_config.json')
fps = 10
recording = None
just_started = True

while True:
    success, depth_frame, rgb_frame, colorized_depth = camera.get_aligned_frame()
    if not success:
        continue

    image_frame = rgb_frame.copy()
    frame_height, frame_width, frame_channel_count = rgb_frame.shape
    image_frame = cv2.resize(image_frame, (frame_width // 2, frame_height // 2))
    cv2.imshow("RGB Frame", image_frame)

    rgb_frame = np.expand_dims(rgb_frame, axis=3)
    depth_frame = np.expand_dims(depth_frame, axis=2)
    depth_frame = np.expand_dims(depth_frame, axis=3)
    new_frame = np.concatenate((rgb_frame, depth_frame), axis=2)

    if just_started:
        recording = new_frame
        just_started = False
    else:
        recording = np.concatenate((recording, new_frame), axis=3)

    time.sleep(1/fps)

    key = cv2.waitKey(1)
    if key == 27:  # 'Esc'
        print("Saving to file...")
        with open('recording.npy', 'wb') as f:
            np.save(f, recording)
        camera.release()
        print("Finished")
        break
