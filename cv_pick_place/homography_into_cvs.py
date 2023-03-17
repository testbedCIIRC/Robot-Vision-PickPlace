import cv2
import os
import sys
import csv
import time

sys.path.append(
    "C:\\Users\\Testbed\\Documents\\robot_cell\\Robot-Vision-PickPlace\\Robot-Vision-PickPlace\\cv_pick_place\\robot_cell\\detection\\"
)

# os.path.abspath("cv_pick_place\robot_cell\detection")
from realsense_depth import DepthCamera
import numpy as np
import json
import argparse
from robot_cell.detection.threshold_detector import ThresholdDetector


def write_to_csv(filename: str, header: list, data: list[list]):
    """Writes data to a csv file.

    Args:
        filename (str): name of the file to write to
        header (list): list of column names
        data (list[list]): list of lists of data to write
    """
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(header)
        writer.writerows(data)


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


# convyor point in marker_id and its points
world_points = {
    "5": [0.0, 0.0],
    "23": [0.0, 14.8],
    "6": [24.8, 4.95],
    "18": [24.8, 14.8],
    "2": [14.9, 0],
    "15": [9.95, 9.9],
}


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


ROB_CONFIG_FILE = os.path.join(
    "C:\\Users\\Testbed\\Documents\\robot_cell\\Robot-Vision-PickPlace\\Robot-Vision-PickPlace\\cv_pick_place\\config",
    "robot_config.json",
)

parser = argparse.ArgumentParser(description="Robot cell input arguments.")
parser.add_argument(
    "--config-file",
    default=ROB_CONFIG_FILE,
    type=str,
    dest="CONFIG_FILE",
    help="Path to configuration file",
)
config, _ = parser.parse_known_args()

with open(config.CONFIG_FILE, "r") as file:
    rob_config = json.load(file)

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

camera = DepthCamera(config_path=rob_config.path_camera_config)

intrinsic = camera.intr
camera_parameter = [intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy]
fx, fy, cx, cy = camera_parameter
K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
dist = np.array(camera.intr.coeffs)
print(K)

# with open(config.CONFIG_FILE, "r") as file:
#     rob_config = json.load(file)

image_points = {}
extrensic_transformation = np.eye(4)
frame = 0

# detector = ThresholdDetector(
#     rob_config.hsv_ignore_vertical,
#     rob_config.hsv_ignore_horizontal,
#     rob_config.hsv_max_ratio_error,
#     rob_config.hsv_white_lower,
#     rob_config.hsv_white_upper,
#     rob_config.hsv_brown_lower,
#     rob_config.hsv_brown_upper,
# )

run_num = 7
csv_file_name = "homo" + str(run_num) + ".csv"
csv_header = [
    "time [s]",
    "h_x [cm]",
    "h_y [cm]",
    "c_x [cm]",
    "c_y [cm]",
    "c_z [cm]",
    "L2 [cm]",
    "L2_x [cm]",
    "L2_y [cm]",
]
csv_data = []  #

save_frame = 5  # 10 # When it should save
frame_num = 0


while True:
    world_points_detect = []
    image_points_detect = []
    marker_centroid = []
    # Get frames from camera
    success, _, depth_frame, rgb_frame, colorized_depth = camera.get_frames()
    if not success:
        continue

    frame_height, frame_width, frame_channel_count = rgb_frame.shape
    text_size = frame_height / 1000

    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        gray_frame,
        cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11),
        parameters=cv2.aruco.DetectorParameters_create(),
    )
    data = []
    for (tag_corners, tag_id) in zip(corners, ids):

        if tag_id == 25:
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
            cv2.circle(rgb_frame, (cX, cY), 10, (0, 255, 0), -1)
            # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
            # tag_corners, 0.162, K, dist)
            # rotation_matrix, jacobian = cv2.Rodrigues(rvec)
            # rotation_angle = rotation_angles(rotation_matrix, "zyx")
            # print("rotation angle", rotation_angle)
            # ax_frame = cv2.drawFrameAxes(rgb_frame, K, dist, rvec, tvec, 0.08)
            # cv2.imshow("Axis", ax_frame)
            marker_centroid = [cX, cY]

        # take any individual marker as packet marker id 1
        if tag_id == 5:
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
            cv2.circle(rgb_frame, (cX, cY), 10, (255, 0, 0), -1)
            marker_centroid_2 = [cX, cY]

            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                tag_corners, 0.033, K, dist
            )
            # ax_frame = cv2.drawFrameAxes(rgb_frame, K, dist, rvec, tvec, 0.08)
            # cv2.imshow("Axis", ax_frame)
            marker_center = camera.pixel_to_3d_point(
                marker_centroid_2, camera.get_raw_depth_frame()
            )
            rotation_matrix, jacobian = cv2.Rodrigues(rvec)
            # rotation_angle = rotation_angles(rotation_matrix, "zyx")
            # print("rotation angle", rotation_angle)
            extrensic_transformation[:3, :3] = rotation_matrix
            # transformation_marker[:3, 3:] = np.array(tvec).reshape(3, 1)
            extrensic_transformation[:3, 3:] = np.array(marker_center).reshape(3, 1)
            # print(" transformation updated")
            # print(tvec)
            # print(marker_center)

        else:
            corners = tag_corners.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners

            top_left = (int(top_left[0]), int(top_left[1]))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

            # Compute centroid
            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)
            image_points[str(int(tag_id))] = [cX, cY]

            # transformation_marker = np.linalg.inv(transformation_marker)

    homography_matrix = np.eye(4)
    frame += 1
    for tag_id in image_points:
        if tag_id in world_points:
            world_points_detect.append(world_points[tag_id])
            image_points_detect.append(image_points[tag_id])
    # print(f"W{world_points_detect}, I{image_points_detect} \n")
    # print(f"W{world_points},\n I{image_points} \n")
    homography_matrix, _ = cv2.findHomography(
        np.array(image_points_detect), np.array(world_points_detect)
    )

    try:
        extrensic_transformation = np.linalg.inv(extrensic_transformation)
        threed_point = np.array(
            camera.pixel_to_3d_point(marker_centroid, camera.get_raw_depth_frame())
        ).reshape(3, 1)
        # print(extrensic_transformation)
        threed_point = np.append(threed_point, 1)
        # print(threed_point)
        transformed_3d_point = extrensic_transformation @ threed_point
        transformed_3d_point = transformed_3d_point * 1000
        twod_point = transformed_3d_point[0:2]

        # print(transformed_3d_point * 1000)

        homo_point = homography_matrix @ np.array(
            [marker_centroid[0], marker_centroid[1], 1]
        )
        homo_2d = homo_point[0:2] * 10

        distance = np.linalg.norm(twod_point - homo_2d)
        distance_x = np.linalg.norm(twod_point[0] - homo_2d[0])
        distance_y = np.linalg.norm(twod_point[1] - homo_2d[1])
        print(
            "distance in x, y and distance ",
            round(distance_x / 10),
            round(distance_y / 10),
            round(distance / 10),
        )
        # print(homo_point.reshape(1, 3)[0]*10)
        csv_header = [
            "time [s]",
            "h_x [cm]",
            "h_y [cm]",
            "c_x [cm]",
            "c_y [cm]",
            "c_z [cm]",
            "L2 [cm]",
            "L2_x [cm]",
            "L2_y [cm]",
        ]
        data = [
            time.time(),
            homo_point[0],
            homo_point[1],
            transformed_3d_point[0] / 10,
            transformed_3d_point[1] / 10,
            transformed_3d_point[2] / 10,
            distance / 10,
            distance_x / 10,
            distance_y / 10,
        ]
    except:
        pass

    cv2.imshow("window_name", rgb_frame)

    # Press Q on keyboard to  exit
    key = cv2.waitKey(1)
    if key in [27, ord("q")]:
        break
    # if cv2.waitKey(25) & 0xFF == ord("q"):a
    #     break

    # print(f"H{marker_centroid}")
    if frame_num % save_frame == 0 and len(data) != 0:
        csv_data.append(data)

    frame_num += 1

write_to_csv(csv_file_name, csv_header, csv_data)

cv2.destroyAllWindows()
