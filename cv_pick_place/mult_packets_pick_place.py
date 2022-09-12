# Standard libraries
import time
import multiprocessing
import multiprocessing.connection
import multiprocessing.managers

# Third party libraries
import cv2
import numpy as np

# Local
from robot_cell.control.control_state_machine import RobotStateMachine
from robot_cell.control.robot_control import RcCommand
from robot_cell.control.robot_control import RcData
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag
from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.packet.packet_object import Packet
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.packet.grip_position_estimation import GripPositionEstimation
from robot_cell.graphics_functions import drawText
from robot_cell.graphics_functions import colorizeDepthFrame
from robot_cell.graphics_functions import show_boot_screen


def packet_tracking(
    pt: ItemTracker,
    detected_packets: list[Packet],
    depth_frame: np.ndarray,
    frame_width: int,
    mask: np.ndarray,
) -> None:
    """
    Assigns IDs to detected packets and updates packet depth frames.

    Args:
        pt (ItemTracker): ItemTracker class.
        detected_packets (list[Packet]): List of detected packets.
        depth_frame (np.ndarray): Depth frame from camera.
        frame_width (int): Width of the camera frame in pixels.
        mask (np.ndarray): Binary mask of packet.

    Returns:
        registered_packets (list[Packet]): List of tracked packet objects.
    """

    # Update tracked packets from detected packets
    labeled_packets = pt.track_items(detected_packets)
    pt.update_item_database(labeled_packets)

    # Update depth frames of tracked packets
    for item in pt.item_database:
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
    registered_packets = pt.item_database

    return registered_packets


def draw_frame(
    image_frame: np.ndarray,
    registered_packets: list[Packet],
    encoder_pos: int,
    text_size: float,
    toggles_dict: dict,
    info_dict: dict,
    colorized_depth: np.ndarray,
    start_time: float,
    frame_width: int,
    frame_height: int,
    resolution: tuple[int, int],
) -> None:
    """
    Draw information on image frame.

    Args:
        image_frame (np.ndarray): Frame into which the information will be draw.
        registered_packets (list[Packet]): List of tracked packet objects.
        encoder_pos (int): Encoder position value.
        text_size (float): Size modifier of the text.
        toggles_dict (dict): Dictionary of variables toggling various drawing functions.
        info_dict (dict): Dictionary of variables containing information about the cell.
        colorized_depth (np.ndarray): Frame containing colorized depth values.
        start_time (float): Start time of current frame. Should be measured at start of every while loop.
        frame_width (int): Width of the camera frame in pixels.
        frame_height (int): Height of the camera frame in pixels.
        resolution (tuple[int, int]): Resolution of the on screen window.
    """

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
                (packet.centroid_px.x + 10, packet.centroid_px.y + int(45 * text_size)),
                text_size,
            )

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
        text_robot = str(info_dict)
        drawText(image_frame, text_robot, (10, int(75 * text_size)), text_size)

    image_frame = cv2.resize(image_frame, resolution)

    # Show frames on cv2 window
    cv2.imshow("Frame", image_frame)


def process_key_input(
    key: int,
    control_pipe: multiprocessing.connection.PipeConnection,
    toggles_dict: dict,
    is_rob_ready: bool,
) -> tuple[bool, dict]:
    """
    Process input from keyboard.

    Args:
        key (int): ID of pressed key.
        control_pipe (multiprocessing.connection.PipeConnection): Multiprocessing pipe object for sending commands to RobotControl object process.
        toggles_dict (dict): Dictionary of toggle variables for indication and control.
        is_rob_ready (bool): Shows if robot is ready.

    Returns:
        end_prog (bool): Indicates if the program should end.
        toggles_dict (dict): Dictionary of toggle variables changed according to user input.
    """

    end_prog = False

    # Toggle gripper
    if key == ord("g"):
        toggles_dict["gripper"] = not toggles_dict["gripper"]
        control_pipe.send(RcData(RcCommand.GRIPPER, toggles_dict["gripper"]))

    # Toggle conveyor in left direction
    if key == ord("n"):
        toggles_dict["conv_left"] = not toggles_dict["conv_left"]
        control_pipe.send(RcData(RcCommand.CONVEYOR_LEFT, toggles_dict["conv_left"]))

    # Toggle conveyor in right direction
    if key == ord("m"):
        toggles_dict["conv_right"] = not toggles_dict["conv_right"]
        control_pipe.send(RcData(RcCommand.CONVEYOR_RIGHT, toggles_dict["conv_right"]))

    # Toggle detected packets bounding box display
    if key == ord("b"):
        toggles_dict["show_bbox"] = not toggles_dict["show_bbox"]

    # Toggle depth map overlay
    if key == ord("d"):
        toggles_dict["show_depth_map"] = not toggles_dict["show_depth_map"]

    # Toggle frame data display
    if key == ord("f"):
        toggles_dict["show_frame_data"] = not toggles_dict["show_frame_data"]

    # Toggle HSV mask overlay
    if key == ord("h"):
        toggles_dict["show_hsv_mask"] = not toggles_dict["show_hsv_mask"]

    # Abort program
    if key == ord("a"):
        control_pipe.send(RcData(RcCommand.ABORT_PROGRAM))

    # Continue program
    if key == ord("c"):
        control_pipe.send(RcData(RcCommand.CONTINUE_PROGRAM))

    # Stop program
    if key == ord("s"):
        control_pipe.send(RcData(RcCommand.STOP_PROGRAM))

    # Print info
    if key == ord("i"):
        print("[INFO]: Is robot ready = {}".format(is_rob_ready))

    if key == 27:  # Esc
        end_prog = True

    return end_prog, toggles_dict


def main_multi_packets(
    rob_config: dict,
    rob_dict: dict,
    info_dict: multiprocessing.managers.DictProxy,
    encoder_pos_m: multiprocessing.managers.ValueProxy,
    control_pipe: multiprocessing.connection.PipeConnection,
) -> None:
    """
    Process for pick and place with moving conveyor and point cloud operations.

    Args:
        rob_config (dict): Dictionary with parameters setting the behaviour of the cell.
        rob_dict (dict): Dictionary of predefined points.
        info_dict (multiprocessing.managers.DictProxy): Dictionary from multiprocessing Manager for reading OPCUA info from another process.
        encoder_pos_m (multiprocessing.managers.ValueProxy): Value object from multiprocessing Manager for reading encoder value from another process.
        control_pipe (multiprocessing.connection.PipeConnection): Multiprocessing pipe object for sending commands to RobotControl object process.
    """

    # Load home position from dictionary
    home_xyz_coords = np.array(
        [
            rob_dict["home_pos"][0]["x"],
            rob_dict["home_pos"][0]["y"],
            rob_dict["home_pos"][0]["z"],
        ]
    )

    # Inititalize objects
    apriltag = ProcessingApriltag()
    apriltag.load_world_points(rob_config.path_homography_points)

    pt = ItemTracker(
        max_disappeared_frames=rob_config.tracker_frames_to_deregister,
        guard=rob_config.tracker_guard,
        max_item_distance=rob_config.tracker_max_item_distance,
    )

    dc = DepthCamera(config_path=rob_config.path_camera_config)

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

    constants = {
        "frame_limit": rob_config.frame_limit,
        "packet_depths": rob_config.packet_depths,
        "min_pick_distance": rob_config.min_pick_distance,
        "max_pick_distance": rob_config.max_pick_distance,
        "z_offset": rob_config.z_offset,
        "x_pick_offset": rob_config.x_pick_offset,
        "grip_time_offset": rob_config.grip_time_offset,
        "pick_start_x_offset": rob_config.pick_start_x_offset,
        "max_z": rob_config.max_z,
        "min_y": rob_config.min_y,
        "max_y": rob_config.max_y,
    }

    stateMachine = RobotStateMachine(
        control_pipe,
        gripper_pose_estimator,
        encoder_pos_m,
        home_xyz_coords,
        constants=constants,
        verbose=rob_config.verbose,
    )

    if rob_config.detector_type == "NN1":
        show_boot_screen("STARTING NEURAL NET...")
        nn1_paths = {
            "ANNOTATION_PATH": rob_config.nn1_annotation_path,
            "CHECKPOINT_PATH": rob_config.nn1_checkpoint_path,
        }
        nn1_files = {
            "PIPELINE_CONFIG": rob_config.nn1_pipeline_config,
            "LABELMAP": rob_config.nn1_labelmap,
        }
        pack_detect = PacketDetector(
            nn1_paths,
            nn1_files,
            rob_config.nn1_checkpoint,
            rob_config.nn1_max_detections,
            rob_config.nn1_detection_threshold,
        )
    elif rob_config.detector_type == "NN2":
        # TODO Implement new deep detector
        pass
    elif rob_config.detector_type == "HSV":
        pack_detect = ThresholdDetector(
            ignore_vertical_px=rob_config.hsv_ignore_vertical,
            ignore_horizontal_px=rob_config.hsv_ignore_horizontal,
            max_ratio_error=rob_config.hsv_max_ratio_error,
            white_lower=rob_config.hsv_white_lower,
            white_upper=rob_config.hsv_white_upper,
            brown_lower=rob_config.hsv_brown_lower,
            brown_upper=rob_config.hsv_brown_upper,
        )

    # Toggles
    toggles_dict = {
        "gripper": False,  # Gripper state
        "conv_left": False,  # Conveyor heading left enable
        "conv_right": False,  # Conveyor heading right enable
        "show_bbox": True,  # Bounding box visualization enable
        "show_frame_data": False,  # Show frame data (robot pos, encoder vel, FPS ...)
        "show_depth_map": False,  # Overlay colorized depth enable
        "show_hsv_mask": False,  # Remove pixels not within HSV mask boundaries
    }

    # Variables
    frame_count = 1  # Counter of frames for homography update
    text_size = 1
    homography = None  # Homography matrix

    # Set home position from dictionary on startup
    control_pipe.send(RcData(RcCommand.SET_HOME_POS_SH))

    while True:
        # Start timer for FPS estimation
        start_time = time.time()

        # READ DATA
        ###################

        # Read data dict from PLC server
        try:
            rob_stopped = info_dict["rob_stopped"]
            stop_active = info_dict["stop_active"]
            prog_done = info_dict["prog_done"]
            encoder_vel = info_dict["encoder_vel"]
            pos = info_dict["pos"]
            speed_override = info_dict["speed_override"]
        except:
            continue

        # Read encoder dict from PLC server
        encoder_pos = encoder_pos_m.value
        if encoder_pos is None:
            continue

        # Get frames from realsense
        success, depth_frame, rgb_frame, colorized_depth = dc.get_frames()
        if not success:
            continue

        frame_height, frame_width, frame_channel_count = rgb_frame.shape
        text_size = frame_height / 1000
        image_frame = rgb_frame.copy()

        # Draw HSV mask over screen if enabled
        if toggles_dict["show_hsv_mask"] and rob_config.detector_type == "HSV":
            image_frame = pack_detect.draw_hsv_mask(image_frame)

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
                pack_detect.set_homography(homography)

        # PACKET DETECTION
        ##################

        # Detect packets using neural network
        if rob_config.detector_type == "NN1":
            image_frame, detected_packets = pack_detect.deep_pack_obj_detector(
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
            pass

        # Detect packets using neural HSV thresholding
        elif rob_config.detector_type == "HSV":
            image_frame, detected_packets, mask = pack_detect.detect_packet_hsv(
                rgb_frame,
                encoder_pos,
                draw_box=toggles_dict["show_bbox"],
                image_frame=image_frame,
            )

        registered_packets = packet_tracking(
            pt, detected_packets, depth_frame, frame_width, mask
        )

        # STATE MACHINE
        ###############

        # Robot ready when programs are fully finished and it isn't moving
        is_rob_ready = prog_done and (rob_stopped or not stop_active)
        stateMachine.run(homography, is_rob_ready, registered_packets, encoder_vel, pos)

        # FRAME GRAPHICS
        ################

        draw_frame(
            image_frame,
            registered_packets,
            encoder_pos,
            text_size,
            toggles_dict,
            info_dict,
            colorized_depth,
            start_time,
            frame_width,
            frame_height,
            (frame_width // 2, frame_height // 2),
        )

        # Keyboard inputs
        key = cv2.waitKey(1)
        end_prog, toggles_dict = process_key_input(
            key, control_pipe, toggles_dict, is_rob_ready
        )

        # End main
        if end_prog:
            control_pipe.send(RcData(RcCommand.CLOSE_PROGRAM))
            cv2.destroyAllWindows()
            dc.release()
            break
