import time
import multiprocessing
import multiprocessing.connection
import multiprocessing.managers

import cv2
import numpy as np

from robot_cell.packet.packet_object import Packet
from robot_cell.control.robot_control import RcCommand
from robot_cell.control.robot_control import RcData
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag
from robot_cell.packet.grip_position_estimation import GripPositionEstimation
from robot_cell.graphics_functions import show_boot_screen
from robot_cell.graphics_functions import drawText
from robot_cell.graphics_functions import colorizeDepthFrame


def compute_mean_packet_z(packet: Packet, pack_z_fixed: float):
    """
    Computes depth of packet based on average of stored depth frames.
    Args:
        packet (Packet): Packet object for which centroid depth should be found.
        pack_z_fixed (float): Constant depth value to fall back to.
    """

    conv2cam_dist = 777.0  # mm
    # range 25 - 13
    depth_mean = packet.avg_depth_crop
    d_rows, d_cols = depth_mean.shape

    # If depth frames are present
    try:
        if d_rows > 0 and d_cols > 0:
            # Compute centroid in depth crop coordinates
            cx, cy = packet.centroid
            xminbbx = packet.xminbbx
            yminbbx = packet.yminbbx
            x_depth, y_depth = int(cx - xminbbx), int(cy - yminbbx)

            # Get centroid from depth mean crop
            centroid_depth = depth_mean[y_depth, x_depth]

            # Compute packet z position with respect to conveyor base
            pack_z = abs(conv2cam_dist - centroid_depth)

            # Return pack_z if in acceptable range, set to default if not
            pack_z_in_range = (pack_z > pack_z_fixed) and (pack_z < pack_z_fixed + 50.0)

            if pack_z_in_range:
                return pack_z
            else:
                return pack_z_fixed

        # When depth frames unavailable
        else:
            return pack_z_fixed

    except:
        return pack_z_fixed


def compute_gripper_rot(angle: float):
    """
    Computes the gripper rotation based on the detected packet angle. For rotating at picking.

    Args:
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


def main_pick_place(
    rob_config: dict,
    rob_dict: dict,
    manag_info_dict: multiprocessing.managers.DictProxy,
    manag_encoder_val: multiprocessing.managers.ValueProxy,
    control_pipe: multiprocessing.connection.PipeConnection,
) -> None:
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

    # Inititalize Apriltag Detector
    apriltag = ProcessingApriltag()
    apriltag.load_world_points(rob_config.path_homography_points)

    # Initialize object tracker
    tracker = ItemTracker(
        rob_config.tracker_frames_to_deregister,
        rob_config.tracker_guard,
        rob_config.tracker_max_item_distance,
    )

    # Initialize depth camera
    camera = DepthCamera(config_path=rob_config.path_camera_config)

    # Initialize gripper pose estimator
    pack_depths = [
        10.0,
        3.0,
        5.0,
        5.0,
    ]  # List of z positions at pick. (TODO, remove when pose estimation is implemented)
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

    # Initialize object detector
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
        print("[INFO] NN1 detector started")
    elif rob_config.detector_type == "NN2":
        show_boot_screen("STARTING NEURAL NET...")
        detector = None  # TODO Implement new deep detector
        print("[INFO] NN2 detector started")
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
        print("[INFO] HSV detector started")
    else:
        detector = None
        print("[WARNING] No detector selected")

    # Tell PLC to use different set of robot instructions
    control_pipe.send(RcData(RcCommand.PICK_PLACE_SELECT, False))

    # Toggles for various program functions
    toggles_dict = {
        "gripper": False,  # Gripper enable
        "conv_left": False,  # Conveyor heading left enable
        "conv_right": False,  # Conveyor heading right enable
        "show_bbox": True,  # Bounding box visualization enable
        "show_frame_data": False,  # Show frame data (robot pos, encoder vel, FPS ...)
        "show_depth_map": False,  # Overlay colorized depth enable
        "show_hsv_mask": False,  # Remove pixels not within HSV mask boundaries
    }

    # Program variables
    frame_count = 1  # Counter of frames for homography update
    text_size = 1
    homography = None  # Homography matrix
    bpressed = 0

    while True:
        # Start timer for FPS estimation
        start_time = time.time()

        # READ DATA
        ###################

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

        # Get frames from camera
        success, depth_frame, rgb_frame, colorized_depth = camera.get_frames()
        if not success:
            continue

        # Crop frames to 1080 x 1440.
        rgb_frame = rgb_frame[:, 240:1680]
        depth_frame = depth_frame[:, 240:1680]
        colorized_depth = colorized_depth[:, 240:1680]

        frame_height, frame_width, frame_channel_count = rgb_frame.shape
        text_size = frame_height / 1000

        # rgb_frame is used for detection, image_frame is used for graphics and displayed
        image_frame = rgb_frame.copy()

        # Draw HSV mask over screen if enabled
        if toggles_dict["show_hsv_mask"] and rob_config.detector_type == "HSV":
            image_frame = detector.draw_hsv_mask(image_frame)

        # HOMOGRAPHY UPDATE
        ###################

        # Update homography
        if frame_count == 1:
            apriltag.detect_tags(rgb_frame)
            homography = apriltag.compute_homog()

        image_frame = apriltag.draw_tags(image_frame)

        # If homography has been detected
        if isinstance(homography, np.ndarray):
            # Increase counter for homography update
            frame_count += 1
            if frame_count >= rob_config.homography_frame_count:
                frame_count = 1

            # Set homography in HSV detector
            if rob_config.detector_type == "HSV":
                detector.set_homography(homography)

        # PACKET DETECTION
        ##################

        # Detect packets using neural network
        if rob_config.detector_type == "NN1":
            image_frame, detected_packets = detector.deep_pack_obj_detector(
                rgb_frame,
                depth_frame,
                encoder_pos,
                bnd_box=toggles_dict["show_bbox"],
                homography=homography,
                image_frame=image_frame,
            )
            for packet in detected_packets:
                packet.width = packet.width * frame_width
                packet.height = packet.height * frame_height

        # Detect packets using neural network
        elif rob_config.detector_type == "NN2":
            # TODO Implement new deep detector
            detected_packets = []

        # Detect packets using HSV thresholding
        elif rob_config.detector_type == "HSV":
            image_frame, detected_packets, mask = detector.detect_packet_hsv(
                rgb_frame,
                encoder_pos,
                toggles_dict["show_bbox"],
                image_frame,
            )

        # In case no valid detector was selected
        else:
            detected_packets = []

        # PACKET TRACKING
        #################

        labeled_packets = tracker.track_items(detected_packets)
        tracker.update_tracked_item_list(labeled_packets, homography, encoder_pos)

        # Update depth frames of tracked packets
        for item in tracker.tracked_item_list:
            if item.disappeared == 0:
                # Check if packet is far enough from edge
                if (
                    item.centroid_px.x - item.width / 2 > item.crop_border_px
                    and item.centroid_px.x + item.width / 2
                    < (frame_width - item.crop_border_px)
                ):
                    depth_crop = item.get_crop_from_frame(depth_frame)
                    mask_crop = item.get_crop_from_frame(mask)
                    item.add_depth_crop_to_average(depth_crop)
                    item.set_mask(mask_crop)

        # Update registered packet list with new packet info
        registered_packets = tracker.tracked_item_list

        # FRAME GRAPHICS
        ################

        # Draw packet info
        for packet in registered_packets:
            if packet.disappeared == 0:
                # Draw centroid estimated with encoder position
                cv2.drawMarker(
                    image_frame,
                    packet.getCentroidFromEncoder(encoder_pos),
                    (255, 255, 0),
                    cv2.MARKER_CROSS,
                    10,
                    cv2.LINE_4,
                )

                # Draw packet ID and type
                text_id = "ID {}, Type {}".format(packet.id, packet.type)
                drawText(
                    image_frame,
                    text_id,
                    (packet.centroid_px.x + 10, packet.centroid_px.y),
                    text_size,
                )

                # Draw packet centroid value in pixels
                text_centroid = "X: {}, Y: {} (px)".format(
                    packet.centroid_px.x, packet.centroid_px.y
                )
                drawText(
                    image_frame,
                    text_centroid,
                    (
                        packet.centroid_px.x + 10,
                        packet.centroid_px.y + int(45 * text_size),
                    ),
                    text_size,
                )

                # Draw packet centroid value in milimeters
                centroid_mm = packet.get_centroid_in_mm()
                text_centroid = "X: {:.2f}, Y: {:.2f} (mm)".format(
                    centroid_mm.x, centroid_mm.y
                )
                drawText(
                    image_frame,
                    text_centroid,
                    (
                        packet.centroid_px.x + 10,
                        packet.centroid_px.y + int(80 * text_size),
                    ),
                    text_size,
                )

        # Draw packet depth crop to separate frame
        cv2.imshow("Depth Crop", np.zeros((500, 500)))
        for packet in registered_packets:
            if packet.avg_depth_crop is not None:
                depth_img = colorizeDepthFrame(packet.avg_depth_crop)
                depth_img = cv2.resize(depth_img, (500, 500))
                cv2.imshow("Depth Crop", depth_img)
                break

        # Show depth frame overlay
        if toggles_dict["show_depth_map"]:
            image_frame = cv2.addWeighted(image_frame, 0.8, colorized_depth, 0.3, 0)

        # Show FPS and robot position data
        if toggles_dict["show_frame_data"]:
            # Draw FPS to screen
            text_fps = "FPS: {:.2f}".format(1.0 / (time.time() - start_time))
            drawText(image_frame, text_fps, (10, int(35 * text_size)), text_size)

            # Draw OPCUA data to screen
            text_robot = str(manag_info_dict)
            drawText(image_frame, text_robot, (10, int(75 * text_size)), text_size)

        image_frame = cv2.resize(image_frame, (frame_width // 2, frame_height // 2))

        # Show frames on cv2 window
        cv2.imshow("Frame", image_frame)

        key = cv2.waitKey(1)

        is_rob_ready = prog_done and (rob_stopped or not stop_active)
        if is_rob_ready and len(registered_packets) >= 1:
            if key == ord("b"):
                bpressed += 1
                # If begin key is held for 30 frames, start program
                if bpressed == 30:
                    packet = registered_packets[0]
                    centroid_mm = packet.get_centroid_in_mm()
                    packet_x = centroid_mm.x
                    packet_y = centroid_mm.y
                    angle = packet.avg_angle_deg
                    packet_type = packet.type
                    gripper_rot = compute_gripper_rot(angle)
                    trajectory_dict = {
                        "x": packet_x,
                        "y": packet_y,
                        "rot": gripper_rot,
                        "packet_type": packet_type,
                        "x_offset": 0.0,
                        "pack_z": compute_mean_packet_z(
                            packet, pack_depths[packet_type]
                        ),
                    }
                    control_pipe.send(
                        RcData(RcCommand.CHANGE_TRAJECTORY, trajectory_dict)
                    )
                    control_pipe.send(RcData(RcCommand.START_PROGRAM, False))
                    print("[INFO] Program started")
                    bpressed = 0
            else:
                bpressed = 0
        else:
            bpressed = 0

        # Toggle gripper
        if key == ord("g"):
            toggles_dict["gripper"] = not toggles_dict["gripper"]
            control_pipe.send(RcData(RcCommand.GRIPPER, toggles_dict["gripper"]))

        # Toggle conveyor in left direction
        elif key == ord("n"):
            toggles_dict["conv_left"] = not toggles_dict["conv_left"]
            control_pipe.send(
                RcData(RcCommand.CONVEYOR_LEFT, toggles_dict["conv_left"])
            )

        # Toggle conveyor in right direction
        elif key == ord("m"):
            toggles_dict["conv_right"] = not toggles_dict["conv_right"]
            control_pipe.send(
                RcData(RcCommand.CONVEYOR_RIGHT, toggles_dict["conv_right"])
            )

        # Toggle depth map overlay
        elif key == ord("d"):
            toggles_dict["show_depth_map"] = not toggles_dict["show_depth_map"]

        # Toggle frame data display
        elif key == ord("f"):
            toggles_dict["show_frame_data"] = not toggles_dict["show_frame_data"]

        # Toggle HSV mask overlay
        elif key == ord("h"):
            toggles_dict["show_hsv_mask"] = not toggles_dict["show_hsv_mask"]

        elif key == ord("a"):
            control_pipe.send(RcData(RcCommand.ABORT_PROGRAM))

        elif key == ord("c"):
            control_pipe.send(RcData(RcCommand.CONTINUE_PROGRAM))

        elif key == ord("s"):
            control_pipe.send(RcData(RcCommand.STOP_PROGRAM))

        elif key == 27:
            control_pipe.send(RcData(RcCommand.CLOSE_PROGRAM))
            cv2.destroyAllWindows()
            camera.release()
            break
