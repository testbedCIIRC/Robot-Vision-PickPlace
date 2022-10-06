import numpy as np
import cv2
import time


recording = []
with open('recording_2022_07_13.npy', 'rb') as f:
    recording = np.load(f)

frame_count = recording.shape[3]
frame_index = 0
fps = 5

while True:
    rgb_frame = recording[:, :, 0:3, frame_index].astype(np.uint8)
    depth_frame = recording[:, :, 3, frame_index].astype(np.uint16)

    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 5))
    depth_frame_hist = clahe.apply(depth_frame.astype(np.uint8))
    cv2_colorized_depth = cv2.applyColorMap(depth_frame_hist, cv2.COLORMAP_JET)

    frame_height, frame_width, frame_channel_count = rgb_frame.shape
    rgb_frame = cv2.resize(rgb_frame, (frame_width // 2, frame_height // 2))
    cv2_colorized_depth = cv2.resize(cv2_colorized_depth, (frame_width // 2, frame_height // 2))
    cv2.imshow("RGB Frame", rgb_frame)
    cv2.imshow("Depth Frame", cv2_colorized_depth)

    frame_index += 1
    time.sleep(1/fps)

    if frame_index >= frame_count:
        frame_index = 0

    key = cv2.waitKey(1)
    if key == 27:  # 'Esc'
        break
