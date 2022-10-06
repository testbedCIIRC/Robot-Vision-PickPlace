import numpy as np
import time
import os

# import cv2

from robot_cell.detection.realsense_depth import DepthCamera

# Create a window
# cv2.namedWindow("Frame")

dc = DepthCamera()

count = -1
while True:
    start_time = time.time()

    # Get frames from recording
    success, depth_frame, rgb_frame, colorized_depth = dc.get_frames()

    if not success:
        print("[WARINING] No camera frames have been read")
        continue

    if (
        rgb_frame.shape[0] != depth_frame.shape[0]
        and rgb_frame.shape[1] != depth_frame.shape[1]
    ):
        print("[WARINING] Camera frames have incompatible shape")
        continue

    frame_height = rgb_frame.shape[0]
    frame_width = rgb_frame.shape[1]

    # Display output image
    # output_frame = cv2.resize(rgb_frame, (960, 540))
    # cv2.imshow("Frame", output_frame)

    # key = cv2.waitKey(10)
    fps = 1.0 / (time.time() - start_time)
    if fps < 5.0:
        count += 1
    text_fps = "FPS: {:.2f}".format(fps)
    print(text_fps, "\t stutters:", count)

    # Stop program
    # if key == 27:
    #     dc.release()
    #     cv2.destroyAllWindows()
    #     break
