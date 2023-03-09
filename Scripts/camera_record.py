from robot_cell.detection.realsense_depth import DepthCamera
import numpy as np
import cv2
import time

camera = DepthCamera(config_path="./config/D435_camera_config.json")
just_started = True
max_num_frames = 100
_, depth_frame, rgb_frame, _ = camera.get_frames()
frame_height, frame_width, frame_channel_count = rgb_frame.shape

recording = np.empty(
    (frame_height, frame_width, frame_channel_count + 1, max_num_frames), dtype=np.uint16
)

depth_frame = np.expand_dims(depth_frame, axis=2)
recording[:, :, :, 0] = np.concatenate((rgb_frame.astype(np.uint16), depth_frame), axis=2)

time.sleep(5)

frm_num = 1
while True:
    start_time = time.time()
    success, depth_frame, rgb_frame, _ = camera.get_frames()
    if not success:
        continue

    cv2.imshow("RGB Frame", rgb_frame)

    depth_frame = np.expand_dims(depth_frame, axis=2)
    recording[:, :, :, frm_num] = np.concatenate((rgb_frame.astype(np.uint16), depth_frame), axis=2)

    end_time = time.time()
    duration = end_time - start_time
    print("Time:", duration, "; Frame number:", frm_num)

    key = cv2.waitKey(1)
    frm_num += 1
    if key == 27 or frm_num >= max_num_frames:  # 'Esc'
        print("Saving to file...")
        with open("recording.npy", "wb") as f:
            np.save(f, recording)
        camera.release()
        print("Finished")
        break
