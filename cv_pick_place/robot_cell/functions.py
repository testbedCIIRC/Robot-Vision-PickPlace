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
                cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2)


def compute_gripper_rot(angle):
    """
    Computes the gripper rotation based on the detected packet angle.

    Parameters:
    angle (float): Detected angle of packet.

    Returns:
    float: Gripper rotation.

    """
    angle = abs(angle)
    if angle > 45:
        rot = 90 + (90 - angle)
    if angle <= 45:
        rot = 90 - angle
    return rot


def show_boot_screen(message, resolution = (960,1280)):
    """
    Opens main frame window with boot screen message.

    Parameters:
    message (str): Message to be displayed.

    """
    boot_screen = np.zeros(resolution)
    cv2.namedWindow('Frame')
    cv2.putText(boot_screen, message, 
                (resolution[0] // 2 - 150, resolution[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Frame", boot_screen)
    cv2.waitKey(1)


def meanFilter(depth_frame):
    kernel = np.ones((10, 10), np.float32) / 25
    filtered_depth_frame = cv2.filter2D(depth_frame, -1, kernel)
    return filtered_depth_frame
    
    
def colorizeDepthFrame(depth_frame):
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 5))
    depth_frame_hist = clahe.apply(depth_frame.astype(np.uint8))
    colorized_depth_frame = cv2.applyColorMap(depth_frame_hist, cv2.COLORMAP_JET)
    return colorized_depth_frame
