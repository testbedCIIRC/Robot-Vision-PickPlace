import numpy as np
import cv2


def drawText(frame, text, position, size = 1):
    cv2.putText(frame, 
                text, 
                position,
                cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4)
    cv2.putText(frame, 
                text, 
                position,
                cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1)


def meanFilter(depth_frame):
    kernel = np.ones((10, 10), np.float32) / 25
    filtered_depth_frame = cv2.filter2D(depth_frame, -1, kernel)
    return filtered_depth_frame
    
    
def colorizeDepthFrame(depth_frame):
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 5))
    depth_frame_hist = clahe.apply(depth_frame.astype(np.uint8))
    colorized_depth_frame = cv2.applyColorMap(depth_frame_hist, cv2.COLORMAP_JET)
    return colorized_depth_frame
