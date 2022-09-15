import os
import time
import multiprocessing
import multiprocessing.connection
import multiprocessing.managers

import cv2
import numpy as np
from opcua import ua

from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag
from robot_cell.packet.grip_position_estimation import GripPositionEstimation
from robot_cell.graphics_functions import show_boot_screen


def main_pick_place(
    rob_config: dict,
    rob_dict: dict,
    manag_info_dict: multiprocessing.managers.DictProxy,
    manag_encoder_val: multiprocessing.managers.ValueProxy,
    control_pipe: multiprocessing.connection.PipeConnection,
):
    """
    Pick and place with static conveyor.
    Object should be placed in front of camera, conveyor belt should not be moving.

    Args:
        rob_config (dict): Dictionary with parameters setting the behaviour of the cell.
        rob_dict (dict): Dictionary of predefined points.
        manag_info_dict (multiprocessing.managers.DictProxy): Dictionary from multiprocessing Manager for reading OPCUA info from another process.
        manag_encoder_val (multiprocessing.managers.ValueProxy): Value object from multiprocessing Manager for reading encoder value from another process.
        control_pipe (multiprocessing.connection.PipeConnection): Multiprocessing pipe object for sending commands to RobotControl object process.
    """

    apriltag = ProcessingApriltag()
    apriltag.load_world_points(os.path.join("config", "conveyor_points.json"))

    tracker = ItemTracker(
        max_disappeared_frames=rob_config.tracker_frames_to_deregister,
        guard=rob_config.tracker_guard,
        max_item_distance=rob_config.tracker_max_item_distance,
    )

    camera = DepthCamera(config_path=rob_config.path_camera_config_demos)

    gripper_pose_estimator = GripPositionEstimation(
        visualize=rob_config.pos_est_visualize,
        verbose=rob_config.verbose,
        center_switch=rob_config.pos_est_center_switch,
        gripper_radius=rob_config.pos_est_gripper_radius,
        gripper_ration=rob_config.pos_est_gripper_ration,
        max_num_tries=rob_config.pos_est_max_num_tries,
        height_th=rob_config.pos_est_height_th,
        num_bins=rob_config.pos_est_num_bins,
        black_list_radius=rob_config.pos_est_blacklist_radius,
        save_depth_array=rob_config.pos_est_save_depth_array,
    )

    # Initialize selested detector
    if rob_config.detector_type == "NN1":
        show_boot_screen("STARTING NEURAL NET...")
        detector = PacketDetector(
            rob_config.nn1_annotation_path,
            rob_config.nn1_checkpoint_path,
            rob_config.nn1_pipeline_config,
            rob_config.nn1_labelmap,
            rob_config.nn1_checkpoint,
            rob_config.nn1_max_detections,
            rob_config.nn1_detection_threshold,
        )
    elif rob_config.detector_type == "NN2":
        show_boot_screen("STARTING NEURAL NET...")
        # TODO Implement new deep detector
        pass
    elif rob_config.detector_type == "HSV":
        detector = ThresholdDetector(
            rob_config.hsv_ignore_vertical,
            rob_config.hsv_ignore_horizontal,
            rob_config.hsv_max_ratio_error,
            rob_config.hsv_white_lower,
            rob_config.hsv_white_upper,
            rob_config.hsv_brown_lower,
            rob_config.hsv_brown_upper,
        )
    else:
        print("[WARNING] No detector selected")

    rc.connect_OPCUA_server()
    rc.get_nodes()
    rc.Pick_Place_Select.set_value(ua.DataValue(False))

    is_detect = False
    conv_left = False
    conv_right = False
    bbox = True
    depth_map = True
    f_data = False
    homography = None
    frame_count = 1

    while True:
        # Start timer for FPS estimation
        start_time = time.time()

        # Read data dict from OPCUA server
        try:
            rob_stopped = manag_info_dict["rob_stopped"]
            stop_active = manag_info_dict["stop_active"]
            prog_done = manag_info_dict["prog_done"]
            encoder_vel = manag_info_dict["encoder_vel"]
            pos = manag_info_dict["pos"]
            speed_override = manag_info_dict["speed_override"]
            encoder_pos = manag_encoder_val.value
            if encoder_pos is None:
                continue
        except:
            continue

        # Get frames from realsense.
        success, depth_frame, rgb_frame, colorized_depth = dc.get_frames()
        height, width, depth = rgb_frame.shape

        # Crop frames to 1080x1440x3.
        rgb_frame = rgb_frame[:, 240:1680]
        depth_frame = depth_frame[:, 240:1680]
        colorized_depth = colorized_depth[:, 240:1680]

        # Update homography
        if frame_count == 1:
            apriltag.detect_tags(rgb_frame)
            homography = apriltag.compute_homog()

        rgb_frame = apriltag.draw_tags(rgb_frame)

        # If homography has been detected
        if isinstance(homography, np.ndarray):
            # Increase counter for homography update
            frame_count += 1
            if frame_count >= 500:
                frame_count = 1

        img_detect, detected = pack_detect.deep_detector(
            rgb_frame, depth_frame, homography, bnd_box=bbox
        )

        objects = ct.update(detected)
        # print(objects)
        rc.objects_update(objects, img_detect)

        if depth_map:
            img_detect = cv2.addWeighted(img_detect, 0.8, colorized_depth, 0.3, 0)

        if f_data:
            cv2.putText(
                img_detect,
                str(info_dict),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                img_detect,
                "FPS:" + str(1.0 / (time.time() - start_time)),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (255, 255, 0),
                2,
            )

        cv2.imshow("Frame", cv2.resize(img_detect, (720, 540)))

        key = cv2.waitKey(1)

        if prog_done and (rob_stopped or not stop_active):
            if key == ord("b"):
                bpressed += 1
                if bpressed == 5:
                    print(detected)
                    world_centroid = detected[0][2]
                    packet_x = round(world_centroid[0] * 10.0, 2)
                    packet_y = round(world_centroid[1] * 10.0, 2)
                    angle = detected[0][3]
                    gripper_rot = rc.compute_gripper_rot(angle)
                    packet_type = detected[0][4]
                    rc.change_trajectory(packet_x, packet_y, gripper_rot, packet_type)
                    rc.start_program()
                    bpressed = 0
            elif key != ord("b"):
                bpressed = 0

        if key == ord("o"):
            rc.Gripper_State.set_value(ua.DataValue(False))
            time.sleep(0.1)

        if key == ord("i"):
            rc.Gripper_State.set_value(ua.DataValue(True))
            time.sleep(0.1)

        if key == ord("m"):
            conv_right = not conv_right
            rc.Conveyor_Right.set_value(ua.DataValue(conv_right))
            time.sleep(0.1)

        if key == ord("n"):
            conv_left = not conv_left
            rc.Conveyor_Left.set_value(ua.DataValue(conv_left))
            time.sleep(0.1)

        if key == ord("l"):
            bbox = not bbox

        if key == ord("h"):
            depth_map = not depth_map

        if key == ord("f"):
            f_data = not f_data

        if key == ord("e"):
            is_detect = not is_detect

        if key == ord("a"):
            rc.Abort_Prog.set_value(ua.DataValue(True))
            print("Program Aborted: ", info_dict["abort"])
            time.sleep(0.5)

        if key == ord("c"):
            rc.Conti_Prog.set_value(ua.DataValue(True))
            print("Continue Program")
            time.sleep(0.5)
            rc.Conti_Prog.set_value(ua.DataValue(False))

        if key == ord("s"):
            rc.Stop_Prog.set_value(ua.DataValue(True))
            print("Program Interrupted")
            time.sleep(0.5)
            rc.Stop_Prog.set_value(ua.DataValue(False))

        if key == 27:
            rc.Abort_Prog.set_value(ua.DataValue(True))
            print("Program Aborted: ", info_dict["abort"])
            rc.Abort_Prog.set_value(ua.DataValue(False))
            rc.client.disconnect()
            cv2.destroyAllWindows()
            print("[INFO]: Client disconnected.")
            time.sleep(0.5)
            break
