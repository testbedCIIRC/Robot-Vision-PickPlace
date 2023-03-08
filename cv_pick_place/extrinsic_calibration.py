import os
import argparse
import sys
# file_path_root = os.path.split(os.path.split(__file__)[0])[0]
# sys.path.append('file_path_root')

from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
ROB_CONFIG_FILE = os.path.join("config", "robot_config.json")
import json
from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag
import numpy as np
import cv2

def rotation_angles(matrix, order):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == 'xzx':
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == 'xyx':
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 *np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == 'xzy':
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == 'xyz':
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == 'yxz':
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == 'yzx':
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == 'zyx':
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == 'zxy':
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)

def drawText(
    frame: np.ndarray, text: str, position: tuple[int, int], size: float = 1
) -> None:
    """
    Draws white text with black border to the frame.

    Args:
        frame (np.ndarray): Frame into which the text will be draw.
        text (str): Text to draw.
        position (tuple[int, int]): Position on the frame in pixels.
        size (float): Size modifier of the text.
    """

    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4)
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2
    )

def bool_str(string: str) -> bool:
    """
    Used in argument parser to detect boolean flags written as a string.

    Args:
        string (str): String to be evaluated.

    Returns:
        bool: True, False, depending on contents of the string.

    Raises:
        argparse.ArgumentTypeError: Error in case the string does not contain any of the expected values.
    """

    if string in ["True", "true"]:
        return True
    elif string in ["False", "false"]:
        return False
    else:
        raise argparse.ArgumentTypeError

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Robot cell input arguments.")
    parser.add_argument(
        "--config-file",
        default=ROB_CONFIG_FILE,
        type=str,
        dest="CONFIG_FILE",
        help="Path to configuration file",
    )
    config, _ = parser.parse_known_args()

    # Read config file using provided path
    with open(config.CONFIG_FILE, "r") as file:
        rob_config = json.load(file)

    # Read all other input arguments as specified inside the config file
    for param in rob_config.items():
        if isinstance(param[1]["default"], bool):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                type=bool_str,
            )
        elif isinstance(param[1]["default"], list):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                nargs=len(param[1]["default"]),
                type=type(param[1]["default"][0]),
            )
        elif isinstance(param[1]["default"], int):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                type=int,
            )
        elif isinstance(param[1]["default"], float):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                type=float,
            )
        elif isinstance(param[1]["default"], str):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                type=str,
            )
        else:
            print(f"[WARNING] Default value of {param[0]} config parameter not handled")
    rob_config = parser.parse_args()

    # Read robot positions dictionaries from json file
    with open(rob_config.path_robot_positions) as file:
        robot_poses = json.load(file)

    # Program variables
    frame_count = 1  # Counter of frames for homography update
    text_size = 1
    homography = None  # Homography matrix

    # Inititalize Apriltag Detector
    apriltag = ProcessingApriltag()
    apriltag.load_world_points(rob_config.path_homography_points)

    detector = ThresholdDetector(
        rob_config.hsv_ignore_vertical,
        rob_config.hsv_ignore_horizontal,
        rob_config.hsv_max_ratio_error,
        rob_config.hsv_white_lower,
        rob_config.hsv_white_upper,
        rob_config.hsv_brown_lower,
        rob_config.hsv_brown_upper,
    )

    camera = DepthCamera(config_path=rob_config.path_camera_config)
    intrinsic = camera.intr
    camera_parameter = [intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy]
    fx, fy, cx, cy = camera_parameter
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    dist = np.array(camera.intr.coeffs)
    print(K)

    file_tran = f = open('extrinsic_matrix.json','r')
    transformation_marker = np.array(json.load(file_tran))

    while True:

        # Get frames from camera
        success, depth_frame, rgb_frame, colorized_depth = camera.get_frames()
        if not success:
            continue


        frame_height, frame_width, frame_channel_count = rgb_frame.shape
        text_size = frame_height / 1000

        # rgb_frame is used for detection, image_frame is used for graphics and displayed
        image_frame = rgb_frame.copy()

        apriltag.detect_tags(rgb_frame)
        homography = apriltag.compute_homog()

        gray_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            gray_frame,
            cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11),
            parameters=cv2.aruco.DetectorParameters_create(),
        )

        # Extrinsics calibration
        """ Use this part to save the transformation matrix from convyor to the camera and use as you use homography,
        use the function 'camera.pixel_to_3d_point(pixel)' to get 3d point along with height, use this height for z axis
         for robot for optimal picking"""
        # TODO: Implement this transformation matrix in your code, give option to user if they want to use this
        # TODO: Homograpy or Extrinsic calibration

        # transformation_marker = np.eye(4)
        #
        # for (tag_corners, tag_id) in zip(corners, ids):
        #
        #     # calibration repect to tag id 1 assumping its origin of convyor
        #     if tag_id == 1:
        #         # Get (x, y) corners of the tag
        #         corners = tag_corners.reshape((4, 2))
        #         (top_left, top_right, bottom_right, bottom_left) = corners
        #
        #         top_left = (int(top_left[0]), int(top_left[1]))
        #         top_right = (int(top_right[0]), int(top_right[1]))
        #         bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        #         bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        #
        #         # Compute centroid
        #         cX = int((top_left[0] + bottom_right[0]) / 2.0)
        #         cY = int((top_left[1] + bottom_right[1]) / 2.0)
        #
        #         cv2.circle(image_frame, (cX,cY), 10, (255, 0, 0), -1)
        #
        #         marker_centroid = [cX, cY]
        #         rvec, tvec, markerPoints =cv2.aruco.estimatePoseSingleMarkers(tag_corners, 0.0325, K, dist)
        #         tdp_marker_center = camera.pixel_to_3d_point(marker_centroid, camera.get_raw_depth_frame())
        #
        #         rotation_matrix, jacobian = cv2.Rodrigues(rvec)
        #
        #         # print angles to see the orientation of marker
        #         # angle_marker_w2c = rotation_angles(rotation_matrix, 'zyx')  # zxy
        #         # print(angle_marker_w2c)
        #
        #         transformation_marker[:3, :3] = rotation_matrix
        #         transformation_marker[:3,3:] = tvec.reshape(3,1)
        #
        #         # convert it from w2c to c2w transformation
        #         transformation_marker = np.linalg.inv(transformation_marker)
        #
        #         # project the centroid of marker to 3d space
        #         marker_point = camera.pixel_to_3d_point(marker_centroid, camera.get_raw_depth_frame())
        #
        #         # TO make sure they corresponds
        #         # print(marker_point)
        #         # print(tvec[0][0])
        #         file =  open('extrinsic_matrix.json', 'w')
        #         json.dump(transformation_marker.tolist(), file, indent=2)
        #         file.close()


        file_tran = f = open('extrinsic_matrix.json','r')
        transformation_marker = np.array(json.load(file_tran))

        image_frame = apriltag.draw_tags(image_frame)
        detector.set_homography(homography)
        image_frame, detected_packets, mask = detector.detect_packet_hsv(rgb_frame,0,True)


        for packet in detected_packets:
            # Draw packet centroid value in milimeters
            text_centroid = "X: {:.2f}, Y: {:.2f} (mm)".format(
                packet.centroid_mm.x, packet.centroid_mm.y
            )
            drawText(
                image_frame,
                text_centroid,
                (packet.centroid_px.x + 10, packet.centroid_px.y + int(80 * text_size)),
                text_size,
            )

            pixel = [packet.centroid_px.x, packet.centroid_px.y]
            threed_point = np.array(camera.pixel_to_3d_point(pixel, camera.get_raw_depth_frame())).reshape(3, 1)
            threed_point = np.append(threed_point, 1)

            transformed_3d_point = np.matmul(transformation_marker, threed_point)
            print(transformed_3d_point * 1000)

        image_frame = cv2.resize(image_frame, (960, 540))
        cv2.imshow('window_name', image_frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # closing all open windows
    cv2.destroyAllWindows()


