import cv2
import numpy as np
import time

from robot_cell.detection.realsense_depth import DepthCamera

def empty_function(x):
    None

# Create a window
cv2.namedWindow('Frame')
cv2.namedWindow('Sliders')

# Create trackbars
cv2.createTrackbar('Min Hue', 'Sliders', 0, 179, empty_function) # Hue is from 0-179 for OpenCV
cv2.createTrackbar('Max Hue', 'Sliders', 0, 179, empty_function)

cv2.createTrackbar('Min Sat', 'Sliders', 0, 255, empty_function)
cv2.createTrackbar('Max Sat', 'Sliders', 0, 255, empty_function)

cv2.createTrackbar('Min Val', 'Sliders', 0, 255, empty_function)
cv2.createTrackbar('Max Val', 'Sliders', 0, 255, empty_function)

cv2.createTrackbar('Frame bounds', 'Sliders', 0, 200, empty_function)

# Set default trackbar values
cv2.setTrackbarPos('Min Hue', 'Sliders', 0)
cv2.setTrackbarPos('Max Hue', 'Sliders', 179)

cv2.setTrackbarPos('Min Sat', 'Sliders', 0)
cv2.setTrackbarPos('Max Sat', 'Sliders', 255)

cv2.setTrackbarPos('Min Val', 'Sliders', 0)
cv2.setTrackbarPos('Max Val', 'Sliders', 255)

cv2.setTrackbarPos('Frame bounds', 'Sliders', 0)

# Initialize trackbar variables
min_h = 0
min_s = 0
min_v = 0
max_h = 0
max_s = 0
max_v = 0

frame_bounds = 0

freeze_frame = False

dc = DepthCamera('recording_2022_05_20.npy', 5)

time.sleep(1)
success, depth_frame, rgb_frame, colorized_depth = dc.get_aligned_frame()

while True:
    # Get frames from recording
    if not freeze_frame:
        success, depth_frame, rgb_frame, colorized_depth = dc.get_aligned_frame()
        
    if not success:
        print("[WARINING] No camera frames have been read")
        continue
        
    if rgb_frame.shape[0] != depth_frame.shape[0] and rgb_frame.shape[1] != depth_frame.shape[1]:
        print("[WARINING] Camera frames have incompatible shape")
        continue
        
    frame_height = rgb_frame.shape[0]
    frame_width = rgb_frame.shape[1]

    # Get current positions of all trackbars
    min_h = cv2.getTrackbarPos('Min Hue', 'Sliders')
    min_s = cv2.getTrackbarPos('Min Sat', 'Sliders')
    min_v = cv2.getTrackbarPos('Min Val', 'Sliders')

    max_h = cv2.getTrackbarPos('Max Hue', 'Sliders')
    max_s = cv2.getTrackbarPos('Max Sat', 'Sliders')
    max_v = cv2.getTrackbarPos('Max Val', 'Sliders')
    
    frame_bounds = cv2.getTrackbarPos('Frame bounds', 'Sliders')

    # Get HSV bounds from trackbar values
    lower_bounds = np.array([min_h, min_s, min_v])
    upper_bounds = np.array([max_h, max_s, max_v])

    # Apply mask to frame
    hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bounds, upper_bounds)
    mask[:frame_bounds, :] = 0
    mask[(frame_height - frame_bounds):, :] = 0
    output_frame = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask)

    # Display output image
    output_frame_1 = cv2.resize(rgb_frame, (frame_width // 2, frame_height // 2))
    output_frame_2 = cv2.resize(output_frame, (frame_width // 2, frame_height // 2))
    cv2.imshow('Frame', np.vstack([output_frame_1, output_frame_2]))

    key = cv2.waitKey(10)

    # Stop program
    if key == 27:
        dc.release()
        cv2.destroyAllWindows()
        break

    # Print trackbar values to terminal
    if key == ord('i'):
        print("Lower HSV bounds: [" + str(min_h) + ", " + str(min_s) + ", " + str(min_v) + "]")
        print("Upper HSV bounds: [" + str(max_h) + ", " + str(max_s) + ", " + str(max_v) + "]")
        print("Frame bounds: " + str(frame_bounds))
        print("------------------")
        
    # Stop camera on current frame
    if key == ord('f'):
        freeze_frame = not freeze_frame
