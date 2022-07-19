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


def show_boot_screen(message, resolution = (540, 960)):
    """
    Opens main frame window with boot screen message.

    Parameters:
    message (str): Message to be displayed.
    resolution (int, int): Resolution of thw window

    """
    boot_screen = np.zeros(resolution)
    cv2.namedWindow('Frame')
    cv2.putText(boot_screen, message, 
                ((resolution[1] // 2) - 150, resolution[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Frame", boot_screen)
    cv2.waitKey(1)


def compute_mean_packet_z(packet, pack_z_fixed):
    """
    Computes depth of packet based on average of stored depth frames.

    Parameters:

    packet (object): Packet object for which centroid depth should be found
    pack_z_fixed (float): Contant depth value to fall back to

    """
    if packet.avg_depth_crop is None:
        print("[WARNING] Avg Depth frame is None")
        return pack_z_fixed
    conv2cam_dist = 777.0 # mm
    # range 25 - 13
    depth_mean = packet.avg_depth_crop
    d_rows, d_cols = depth_mean.shape  
    
    # If depth frames are present.
    try:
        if d_rows > 0:
            # Get centroid from depth mean crop.
            centroid_depth = depth_mean[d_rows // 2, d_cols // 2]

            # Compute packet z position with respect to conveyor base.
            pack_z = abs(conv2cam_dist - centroid_depth) - 8

            # Return pack_z if in acceptable range, set to default if not.
            if pack_z < pack_z_fixed:
                pack_z = pack_z_fixed
            elif pack_z > pack_z_fixed + 20.0:
                pack_z = pack_z_fixed + 20.0

            return pack_z

        # When depth frames unavailable.
        else:
            return pack_z_fixed
    
    except:
        return pack_z_fixed


def meanFilter(depth_frame):
    kernel = np.ones((10, 10), np.float32) / 25
    filtered_depth_frame = cv2.filter2D(depth_frame, -1, kernel)
    return filtered_depth_frame
    
    
def colorizeDepthFrame(depth_frame):
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 5))
    depth_frame_hist = clahe.apply(depth_frame.astype(np.uint8))
    colorized_depth_frame = cv2.applyColorMap(depth_frame_hist, cv2.COLORMAP_JET)
    return colorized_depth_frame
