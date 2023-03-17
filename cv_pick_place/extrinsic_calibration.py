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

target_depth = 1
best_depth_error = None


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

    if order == "xzx":
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == "xyx":
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == "yxy":
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == "yzy":
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == "zyz":
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == "zxz":
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == "xzy":
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == "xyz":
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == "yxz":
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == "yzx":
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == "zyx":
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == "zxy":
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)


def Rz(angle: float):
    """
    Rotation matrix around z-axis
    Args:
        angle (float): Rotation angle in radians
    Returns:
        np.ndarray: Rotation matrix in 3x3 array
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def Ry(angle: float):
    """
    Rotation matrix around y-axis
    Args:
        angle (float): Rotation angle in radians
    Returns:
        np.ndarray: Rotation matrix in 3x3 array
    """
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def Rx(angle: float):
    """
    Rotation matrix around x-axis
    Args:
        angle (float): Rotation angle in radians
    Returns:
        np.ndarray: Rotation matrix in 3x3 array
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def R_from_eulers(z, y, x):
    z = np.deg2rad(z)
    y = np.deg2rad(y)
    x = np.deg2rad(x)
    return Rz(x) @ Ry(y) @ Rx(z)


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


if __name__ == "__main__":

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
    marker_centroid_2 = [0, 0]

    marker_centroid_points = {"4": [], "5": [], "6": [], "7": []}
    marker_homo_points = {"4": [], "5": [], "6": [], "7": []}
    camera_point_dict = {"4": [], "5": [], "6": [], "7": []}
    best_total_error = np.inf
    while True:

        # Get frames from camera
        success, _, depth_frame, rgb_frame, colorized_depth = camera.get_frames()
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

        calibrate = True
        transformation_marker = np.eye(4)

        if calibrate:
            tdp_marker_center = 0
            for (tag_corners, tag_id) in zip(corners, ids):

                # calibration repect to tag id 1 assumping its origin of convyor
                if tag_id == 1:
                    # Get (x, y) corners of the tag
                    corners = tag_corners.reshape((4, 2))
                    (top_left, top_right, bottom_right, bottom_left) = corners

                    top_left = (int(top_left[0]), int(top_left[1]))
                    top_right = (int(top_right[0]), int(top_right[1]))
                    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                    bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

                    # Compute centroid
                    cX = int((top_left[0] + bottom_right[0]) / 2.0)
                    cY = int((top_left[1] + bottom_right[1]) / 2.0)

                    marker_centroid = [cX, cY]
                    # # rvec, tvec, markerPoints =cv2.aruco.estimatePoseSingleMarkers(tag_corners, 0.0332, K, dist)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                        tag_corners, 0.0279, K, dist
                    )

                    tdp_marker_center = camera.pixel_to_3d_point(
                        marker_centroid, camera.get_raw_depth_frame()
                    )
                    # rotation_matrix, jacobian = cv2.Rodrigues(rvec)
                    # transformation_marker[:3, :3] = rotation_matrix
                    transformation_marker[:3, 3:] = np.array(tvec).reshape(3, 1)
                    # transformation_marker[:3, 3:] = np.array(tdp_marker_center).reshape(
                    #     3, 1)
                    # print("tvec of marker 1", tvec)
                    # print("3d point of marker", tdp_marker_center)
                    # print(transformation_marker)

                if tag_id == 2:
                    # rvec, tvec, markerPoints =cv2.aruco.estimatePoseSingleMarkers(tag_corners, 0.0332, K, dist)
                    corners = tag_corners.reshape((4, 2))
                    (top_left, top_right, bottom_right, bottom_left) = corners

                    top_left = (int(top_left[0]), int(top_left[1]))
                    top_right = (int(top_right[0]), int(top_right[1]))
                    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                    bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

                    # Compute centroid
                    cX = int((top_left[0] + bottom_right[0]) / 2.0)
                    cY = int((top_left[1] + bottom_right[1]) / 2.0)

                    marker_centroid_2 = [cX, cY]
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                        tag_corners, 0.1606, K, dist
                    )

                    rotation_matrix, jacobian = cv2.Rodrigues(rvec)
                    rotation_angle = rotation_angles(rotation_matrix, "zyx")
                    print("rotation angle is ", rotation_angle)

                    # rtx = np.copy(rotation_matrix)
                    # rotation_matrix = R_from_eulers(*rotation_angle)
                    # print("rota", rotation_angles(rotation_matrix, "zyx"))
                    # print(rotation_matrix - rtx)

                    transformation_marker[:3, :3] = rotation_matrix
                    # print(transformation_marker)

                if str(tag_id[0]) in marker_centroid_points.keys():
                    corners = tag_corners.reshape((4, 2))
                    (top_left, top_right, bottom_right, bottom_left) = corners

                    top_left = (int(top_left[0]), int(top_left[1]))
                    top_right = (int(top_right[0]), int(top_right[1]))
                    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                    bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

                    # Compute centroid
                    cX = int((top_left[0] + bottom_right[0]) / 2.0)
                    cY = int((top_left[1] + bottom_right[1]) / 2.0)
                    marker_centroid_points[str(tag_id[0])] = [cX, cY]
                    cv2.circle(rgb_frame, (cX, cY), 10, (0, 255, 0), -1)

                # else:
                #     corners = tag_corners.reshape((4, 2))
                #     (top_left, top_right, bottom_right, bottom_left) = corners

                #     top_left = (int(top_left[0]), int(top_left[1]))
                #     top_right = (int(top_right[0]), int(top_right[1]))
                #     bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                #     bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

                #     # Compute centroid
                #     cX = int((top_left[0] + bottom_right[0]) / 2.0)
                #     cY = int((top_left[1] + bottom_right[1]) / 2.0)
                #     cv2.circle(rgb_frame,(cX, cY), 10, (255,0,0), -1 )

        transformation_marker = np.linalg.inv(transformation_marker)
        # # file_tran = f = open('extrinsic_matrix.json','r')
        # # transformation_marker = np.array(json.load(file_tran))

        image_frame = apriltag.draw_tags(image_frame)
        detector.set_homography(homography)
        image_frame, detected_packets, mask = detector.detect_packet_hsv(
            rgb_frame, 0, True
        )

        # for packet in detected_packets:
        #     # Draw packet centroid value in milimeters
        #     text_centroid = "X: {:.2f}, Y: {:.2f} (mm)".format(
        #         packet.get_centroid_in_mm().x, packet.get_centroid_in_mm().y
        #     )
        #     drawText(
        #         image_frame,
        #         text_centroid,
        #         (
        #             packet.get_centroid_in_px().x + 10,
        #             packet.get_centroid_in_px().y + int(80 * text_size),
        #         ),
        #         text_size,
        #     )

        # pixel = [packet.get_centroid_in_px().x, packet.get_centroid_in_px().y]
        raw_depth_frame = camera.get_raw_depth_frame()
        total_distance_error = 0
        offset_x = 0
        offset_y = 1
        print(transformation_marker)
        for tag_id in marker_centroid_points:
            distance_error = 0
            distance_error_x = 0
            distance_error_y = 0
            centroid_pixel = np.array(
                [
                    marker_centroid_points[tag_id][0],
                    marker_centroid_points[tag_id][1],
                    1,
                ]
            ).reshape(3, 1)

            homo_point = homography @ centroid_pixel
            marker_homo_points[tag_id] = homo_point.reshape(1, 3)[0][0:2]

            threed_point = np.array(
                camera.pixel_to_3d_point(centroid_pixel[0:2], raw_depth_frame)
            ).reshape(3, 1)
            threed_point = np.append(threed_point, 1)
            transformed_3d_point = np.matmul(transformation_marker, threed_point)
            # scaling to cms
            transformed_3d_point = transformed_3d_point[0:2] * 100
            # offset in x
            transformed_3d_point[0] = transformed_3d_point[0] - offset_x
            # offset in y
            transformed_3d_point[1] = transformed_3d_point[1] - offset_y

            camera_point_dict[tag_id] = transformed_3d_point

            distance_error = np.linalg.norm(
                transformed_3d_point - homo_point.reshape(1, 3)[0][0:2]
            )

            distance_error_x = np.linalg.norm(
                transformed_3d_point[0] - homo_point.reshape(1, 3)[0][0]
            )

            distance_error_y = np.linalg.norm(
                transformed_3d_point[1] - homo_point.reshape(1, 3)[0][1]
            )
            total_distance_error += distance_error
            # print("distance error of ", tag_id, "is ", distance_error)
            print(
                f"distance_err of {tag_id}:\t X:{distance_error_x:3.2f}\t Y:{distance_error_y:3.2f}\t D:{distance_error:3.2f}"
            )
            # print("distance error in x", distance_error_x)
            # print("distance error in y", distance_error_y)

        # print('homo points ', marker_homo_points['4'])
        # print('camera points ', camera_point_dict['4'])
        print("total distance error", total_distance_error)

        if total_distance_error < best_total_error:
            best_total_error = total_distance_error
            print("matrix saved")
            file = open("extrinsic_matrix.json", "w")
            json.dump(transformation_marker.tolist(), file, indent=2)
            file.close()

        # if total_distance_error > 0 and (
        #     best_depth_error is None or depth_error < best_depth_error
        # ):
        #     best_depth_error = depth_error
        #     print("matrix saved")
        #     file = open("extrinsic_matrix.json", "w")
        #     json.dump(transformation_marker.tolist(), file, indent=2)
        #     file.close()

        # transformed_3d_point = np.matmul(transformation_marker, threed_point)
        # print(transformed_3d_point * 1000)

        # # homo_point = [packet.get_centroid_in_mm().x, packet.get_centroid_in_mm().y]

        # transformed_3d_point_2d = transformed_3d_point[0:2]

        # ## distance between homo and 3d point
        # distance = np.linalg.norm(transformed_3d_point_2d - homo_point)
        # packet_height = transformed_3d_point[2] * 1000
        # #print(packet_height)
        # # print(transformed_3d_point_2d)
        # print(f"Depth: {packet_height:.2f}, Distance: {distance:.2f}")

        # depth_error = np.abs(target_depth - packet_height)
        # if packet_height > 0 and (
        #     best_depth_error is None or depth_error < best_depth_error
        # ):
        #     best_depth_error = depth_error
        #     print("matrix saved")
        #     file = open("extrinsic_matrix.json", "w")
        #     json.dump(transformation_marker.tolist(), file, indent=2)
        #     file.close()

        # find the difference between homoggraphy and the april points

        image_frame = cv2.resize(image_frame, (960, 540))
        cv2.imshow("window_name", image_frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            print(best_total_error)
            break

        # if cv2.waitKey(25) & 0xFF == ord("s"):
        #     file = open("extrinsic_matrix.json", "w")
        #     json.dump(transformation_marker.tolist(), file, indent=2)
        #     file.close()

        # closing all open windows
    cv2.destroyAllWindows()
