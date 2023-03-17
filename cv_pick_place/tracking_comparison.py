import os
import csv
import numpy as np
import cv2
import time
import json
import argparse
import sys

sys.path.append("..")

from opcua import Client
from multiprocessing import Manager

from cv_pick_place.robot_cell.detection.realsense_depth import DepthCamera
from cv_pick_place.robot_cell.detection.apriltag_detection import ProcessingApriltag
from cv_pick_place.robot_cell.detection.threshold_detector import ThresholdDetector
from cv_pick_place.robot_cell.detection.apriltag_detection import ProcessingApriltag
from cv_pick_place.robot_cell.detection.threshold_detector import ThresholdDetector
from cv_pick_place.robot_cell.packet.packet_object import Packet
from cv_pick_place.robot_cell.packet.item_tracker import ItemTracker

# from robot_cell.control.robot_control import RobotControl
from robot_cell.control.robot_communication import RobotCommunication

from cv_pick_place.robot_cell.graphics_functions import drawText
from cv_pick_place.robot_cell.graphics_functions import colorizeDepthFrame
from cv_pick_place.robot_cell.graphics_functions import show_boot_screen

from cv_pick_place.mult_packets_pick_place import packet_tracking

LINE_CLEAR = "\x1b[2K"  # <-- ANSI sequence
PERIOD = 1  # in seconds
MM2CM = 0.1


class RobotControl:
    def __init__(self) -> None:
        pass

    def get_nodes(self) -> None:
        self.encoder_pos = self.client.get_node('ns=3;s="Encoder_1".ActualPosition')

    def connect2OPCUA(self) -> None:
        print("[INFO]: Connecting to OPCUA server")
        password = "CIIRC"
        self.client = Client("opc.tcp://user:" + str(password) + "@10.35.91.101:4840/")
        self.client.connect()
        print("[INFO]: Client connected")

    def disconnect(self) -> None:
        self.client.disconnect()
        print("[INFO]: OPCUA Client disconnected")


ROB_CONFIG_FILE = os.path.join("config", "robot_config.json")


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


def write_to_csv(filename: str, header: list, data: list[list]) -> None:
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


if __name__ == "__main__":
    csv_file_name = "friday2"

    header = ["time", "encoder_position", "h_x", "h_y", "c_x", "c_y", "c_z", "L2", "ID"]
    csv_data_all = []
    csv_data_sel = []
    csv_data_per = []
    csv_data_frame = []

    frame_num = 0
    save_frame = 5
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

    apriltag = ProcessingApriltag()
    apriltag.load_world_points(rob_config.path_homography_points)
    # apriltag.load_world_points("config/conveyor_points.json")
    tracker = ItemTracker(
        rob_config.tracker_frames_to_deregister,
        rob_config.tracker_guard,
        rob_config.tracker_max_item_distance,
    )

    ctrl = RobotControl()
    ctrl.connect2OPCUA()
    ctrl.get_nodes()
    time.sleep(0.5)
    camera = DepthCamera(config_path=rob_config.path_camera_config)
    print("HGI")
    detector = ThresholdDetector(
        rob_config.hsv_ignore_vertical,
        rob_config.hsv_ignore_horizontal,
        rob_config.hsv_max_ratio_error,
        rob_config.hsv_white_lower,
        rob_config.hsv_white_upper,
        rob_config.hsv_brown_lower,
        rob_config.hsv_brown_upper,
    )
    print("[INFO] HSV detector started")

    frame_count = 1  # Counter of frames for homography update
    text_size = 1
    homography = None  # Homography matrix
    key = 0  # Variable to store a keypress

    (
        success,
        timestamp,
        depth_frame,
        rgb_frame,
        colorized_depth,
    ) = camera.get_frames()

    cv2.imshow("Frame", rgb_frame)
    t1 = time.time()
    with Manager() as manager:
        manag_info_dict = manager.dict()
        manag_encoder_val = manager.Value("d", None)
        encoder_origin = ctrl.encoder_pos.get_value()
        while True:
            (
                success,
                timestamp,
                depth_frame,
                rgb_frame,
                colorized_depth,
            ) = camera.get_frames()

            try:
                encoder_pos = ctrl.encoder_pos.get_value()
                print(f"ENCODER POS: {encoder_pos}", end="\r")
                if encoder_pos is None:
                    print(" SKIPPiNG due to encoder")
                    continue
            except:
                continue

            if not success:
                continue

            frame_h, frame_w, _ = rgb_frame.shape
            text_size = frame_w / 1000
            img_frame = rgb_frame.copy()

            if frame_count == 1:
                apriltag.detect_tags(rgb_frame)
                homography = apriltag.compute_homog()

            image_frame = apriltag.draw_tags(img_frame)

            if isinstance(homography, np.ndarray):
                # Increase counter for homography update
                frame_count += 1
                if frame_count >= rob_config.homography_frame_count:
                    frame_count = 1

                # Set homography in HSV detector
                if rob_config.detector_type == "HSV":
                    detector.set_homography(homography)

            image_frame, detected_packets, mask = detector.detect_packet_hsv(
                rgb_frame, encoder_pos, False, image_frame
            )

            registered_packets = packet_tracking(
                tracker,
                detected_packets,
                depth_frame,
                frame_w,
                mask,
                encoder_pos,
            )

            for packet in registered_packets:
                if packet.disappeared == 0:
                    centroid_px = packet.get_centroid_in_px()
                    centroid_mm = packet.get_centroid_in_mm()
                    x, y, z = camera.pixel_to_3d_conveyor_frame(centroid_px)
                    y_err = -0.0025 * x - 0.06 * y + 5
                    y += y_err

                    packet.camera_centroid_x = x
                    packet.camera_base_centroid_x = x
                    packet.camera_centroid_y = y
                    packet.camera_centroid_z = z
                    t_now = time.time()

                    camera_text = f"Camera: X {x:.2f}, Y: {y:.2f}, Z: {z:.2f} (mm)"
                    avg_pos = np.array([x, y])

                    packet_pos = np.array([centroid_mm.x, centroid_mm.y])
                    dst = np.linalg.norm(avg_pos - packet_pos)

                    new_data = [
                        t_now,
                        encoder_pos * MM2CM,
                        centroid_mm.x * MM2CM,
                        centroid_mm.y * MM2CM,
                        x * MM2CM,
                        y * MM2CM,
                        z * MM2CM,
                        dst * MM2CM,
                        packet.id,
                    ]
                    csv_data_all.append(new_data)

                    if frame_num % save_frame == 0:
                        csv_data_frame.append(new_data)

                    if t_now - t1 > PERIOD:
                        t1 = t_now
                        csv_data_per.append(new_data)
                        print(f"Saving per")

                    text2save = (
                        "h_C_X: "
                        + str(centroid_mm.x)
                        + "h_C_Y: "
                        + str(centroid_mm.y)
                        + "GT_C_X: "
                        + str(avg_pos[0])
                        + "GT_C_Y: "
                        + str(avg_pos[1])
                        + "Norm: "
                        + str(dst)
                        + "\n"
                    )
                    drawText(
                        image_frame,
                        camera_text,
                        (
                            centroid_px.x + 10,
                            centroid_px.y + int(-45 * text_size),
                        ),
                        text_size,
                    )
                    homo_text = f"Homography {packet_pos[0]:.2f}, {packet_pos[1]:.2f}"
                    drawText(
                        image_frame,
                        homo_text,
                        (
                            centroid_px.x + 10,
                            centroid_px.y + int(10 * text_size),
                        ),
                        text_size,
                    )

                    if key == ord("s"):
                        print(end=LINE_CLEAR)
                        print("[INFO]: Saving the picture and data to file")
                        t = time.time()
                        cv2.imwrite(str(t) + ".jpg", img_frame)
                        with open("measurment.txt", "a") as f:
                            f.write(text2save)
                    elif key == ord("p"):
                        print(end=LINE_CLEAR)
                        print("[INFO]: Adding position data")
                        csv_data_sel.append(new_data)

            cv2.imshow("Frame", image_frame)
            key = cv2.waitKey(1)
            if key in [ord("q"), 27]:  # 'q' or 'Esc'
                break

        # Writing to CSV:
        write_to_csv(csv_file_name + "_sel.csv", header, csv_data_sel)
        write_to_csv(csv_file_name + "_all.csv", header, csv_data_all)
        write_to_csv(csv_file_name + "_per.csv", header, csv_data_per)
        write_to_csv(csv_file_name + "_frm.csv", header, csv_data_frame)

        cv2.destroyAllWindows()
        ctrl.disconnect()
