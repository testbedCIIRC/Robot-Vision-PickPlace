import cv2 
import time
import numpy as np

from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.control.control_state_machine import RobotStateMachine

from robot_cell.control.robot_control import RcCommand
from robot_cell.control.robot_control import RcData
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag

from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.functions import *

from robot_cell.packet.grip_position_estimation import GripPositionEstimation

# DETECTOR_TYPE = 'deep_1'
# DETECTOR_TYPE = 'deep_2'
DETECTOR_TYPE = 'hsv'

# CONSTANTS
MAX_FRAME_COUNT = 500 # Number of frames between homography updates
PACK_DEPTHS = [10.0, 3.0, 5.0, 5.0] # Predefined packet depths, index corresponds to type of packet.
 # constants for pick place operation
PP_CONSTS = {
                'FRAMES_LIM' : 10, # Max frames object must be tracked to start pick & place
                'PACK_DEPTHS' : PACK_DEPTHS, # Predefined packet depths, index corresponds to type of packet.
                'MIN_PICK_DISTANCE' : 600,  # Minimal x position in mm for packet picking
                'MAX_PICK_DISTANCE' : 1900, # Maximal x position in mm for packet picking
                'Z_OFFSET' : 50.0, # Z height offset from pick height for all positions except for pick position
                'X_PICK_OFFSET' : 140, # X offset between prepick and pick position
                'GRIP_TIME_OFFSET' : 400,  # X offset from current packet position to prepick position
                'PICK_START_X_OFFSET' : 25, # Offset between robot and packet for starting the pick move
                'MAX_Z' : 500,
                'MIN_Y' : 45.0,
                'MAX_Y' : 470.0
            }


def packet_tracking(pt, detected_packets, depth_frame, frame_width, mask):
    """
    Assign IDs to detected packets and update depth frames

    Args:
        pt (class): ItemTracker class
        detected_packets (list): list of detected packets
        depth_frame (numpy.ndarray): Depth frame.
        frame_width (int): width of the camera frame
        mask (np.ndarray): Binary mask of packet

    Returns:
        registered_packets (list[packet_object]): List of tracked packet objects
    """
    # Update tracked packets from detected packets
    labeled_packets = pt.track_items(detected_packets)
    pt.update_item_database(labeled_packets)

    # Update depth frames of tracked packets
    for item in pt.item_database:
        if item.disappeared == 0:
            # Check if packet is far enough from edge
            if item.centroid_px.x - item.width / 2 > item.crop_border_px and item.centroid_px.x + item.width / 2 < (frame_width - item.crop_border_px):
                depth_crop = item.get_crop_from_frame(depth_frame)
                mask_crop = item.get_crop_from_frame(mask)
                item.add_depth_crop_to_average(depth_crop)
                item.set_mask(mask_crop)

    # Update registered packet list with new packet info
    registered_packets = pt.item_database

    return registered_packets


def drawText(frame, text, position, size = 1):
    cv2.putText(frame, 
                text, 
                position,
                cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4)
    cv2.putText(frame, 
                text, 
                position,
                cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2)


def draw_frame(image_frame, registered_packets, encoder_pos, text_size, toggles_dict, info_dict, colorized_depth, start_time, frame_width, frame_height):
    """
    Draw information on image frame

    """
    # Draw packet info
    for packet in registered_packets:
        if packet.disappeared == 0:
            # Draw centroid estimated with encoder position
            cv2.drawMarker(image_frame, packet.getCentroidFromEncoder(encoder_pos), 
                    (255, 255, 0), cv2.MARKER_CROSS, 10, cv2.LINE_4)


            # Draw packet ID and type
            text_id = "ID {}, Type {}".format(packet.id, packet.type)
            drawText(image_frame, text_id, (packet.centroid_px.x + 10, packet.centroid_px.y), text_size)

            # Draw packet centroid value in pixels
            text_centroid = "X: {}, Y: {} (px)".format(packet.centroid_px.x, packet.centroid_px.y)
            drawText(image_frame, text_centroid, (packet.centroid_px.x + 10, packet.centroid_px.y + int(45 * text_size)), text_size)

            # Draw packet centroid value in milimeters
            text_centroid = "X: {:.2f}, Y: {:.2f} (mm)".format(packet.centroid_mm.x, packet.centroid_mm.y)
            drawText(image_frame, text_centroid, (packet.centroid_px.x + 10, packet.centroid_px.y + int(80 * text_size)), text_size)

            packet_depth_mm = compute_mean_packet_z(packet, PACK_DEPTHS[packet.type])
            # Draw packet depth value in milimeters
            text_centroid = "Z: {:.2f} (mm)".format(packet_depth_mm)
            drawText(image_frame, text_centroid, (packet.centroid_px.x + 10, packet.centroid_px.y + int(115 * text_size)), text_size)

    # Draw packet depth crop to separate frame
    cv2.imshow("Depth Crop", np.zeros((500, 500)))
    for packet in registered_packets:
        if packet.avg_depth_crop is not None:
            depth_img = colorizeDepthFrame(packet.avg_depth_crop)
            depth_img = cv2.resize(depth_img, (500, 500))
            cv2.imshow("Depth Crop", depth_img)
            break

    # Show depth frame overlay.
    if toggles_dict['show_depth_map']:
        image_frame = cv2.addWeighted(image_frame, 0.8, colorized_depth, 0.3, 0)

    # Show FPS and robot position data
    if toggles_dict['show_frame_data']:
        # Draw FPS to screen
        text_fps = "FPS: {:.2f}".format(1.0 / (time.time() - start_time))
        drawText(image_frame, text_fps, (10, int(35 * text_size)), text_size)

        # Draw OPCUA data to screen
        text_robot = str(info_dict)
        drawText(image_frame, text_robot, (10, int(75 * text_size)), text_size)

    image_frame = cv2.resize(image_frame, (frame_width // 2, frame_height // 2))

    # Show frames on cv2 window
    cv2.imshow("Frame", image_frame)


def process_key_input(key, control_pipe, toggles_dict, is_rob_ready):
    """
    Process input from keyboard

    Args:
        key (int): pressed key
        control_pipe (Pipe): Communication pipe between processes
        toggles_dict (dict): Dictionary of toggle variables for indication and controll
        is_rob_ready (bool): Shows if robot is ready
    Returns:
        end_prog (bool): End programm
        toggles_dict (dict): Dictionary of toggle variables for indication and controll
    """
    end_prog = False
    # Toggle gripper
    if key == ord('g'):
        toggles_dict['gripper'] = not toggles_dict['gripper']
        control_pipe.send(RcData(RcCommand.GRIPPER, toggles_dict['gripper']))

    # Toggle conveyor in left direction
    if key == ord('n'):
        toggles_dict['conv_left'] = not toggles_dict['conv_left']
        control_pipe.send(RcData(RcCommand.CONVEYOR_LEFT, toggles_dict['conv_left']))

    # Toggle conveyor in right direction
    if key == ord('m'):
        toggles_dict['conv_right'] = not toggles_dict['conv_right']
        control_pipe.send(RcData(RcCommand.CONVEYOR_RIGHT, toggles_dict['conv_right']))

    # Toggle detected packets bounding box display
    if key == ord('b'):
        toggles_dict['show_bbox'] = not toggles_dict['show_bbox']
    
    # Toggle depth map overlay
    if key == ord('d'):
        toggles_dict['show_depth_map'] = not toggles_dict['show_depth_map']

    # Toggle frame data display
    if key == ord('f'):
        toggles_dict['show_frame_data'] = not toggles_dict['show_frame_data']

    # Toggle HSV mask overlay
    if key == ord ('h'):
        toggles_dict['show_hsv_mask'] = not toggles_dict['show_hsv_mask']

    # Abort program
    if key == ord('a'):
        control_pipe.send(RcData(RcCommand.ABORT_PROGRAM))
    
    # Continue program
    if key == ord('c'):
        control_pipe.send(RcData(RcCommand.CONTINUE_PROGRAM))
    
    # Stop program
    if key == ord('s'):
        control_pipe.send(RcData(RcCommand.STOP_PROGRAM))

    # Print info
    if key == ord('i'):
        print("[INFO]: Is robot ready = {}".format(is_rob_ready))

    if key == 27:   # Esc
        end_prog = True

    return end_prog, toggles_dict


def main_multi_packets(rob_dict, paths, files, check_point, info_dict, encoder_pos_m, control_pipe):
    """
    Process for pick and place with moving conveyor and point cloud operations.
    
    Parameters:
    rob_dict (dict): RobotControl object for program execution
    paths (dict): Deep detector parameter
    files (dict): Deep detector parameter
    check_point (string): Deep detector parameter
    info_dict (multiprocessing.dcit): Dict from multiprocessing Manager for reading OPCUA info from another process
    encoder_pos_m (multiprocessing.value): Value object from multiprocessing Manager for reading encoder value from another process
    control_pipe (multiprocessing.pipe): Pipe object connected to another process for sending commands
    
    """
    # Inititalize objects
    apriltag = ProcessingApriltag()
    apriltag.load_world_points('conveyor_points.json')
    pt = ItemTracker(max_disappeared_frames = 20, guard = 50, max_item_distance = 400)
    dc = DepthCamera(config_path = 'D435_camera_config.json')
    
    gripper_pose_estimator = GripPositionEstimation(
        visualize=False, verbose=True, center_switch="mass",
        gripper_radius=0.08, max_num_tries = 100, height_th= -0.76, num_bins=20,
        black_list_radius = 0.01, save_depth_array=False
    )
    # Load home position from dictionary
    home_xyz_coords = np.array([rob_dict['home_pos'][0]['x'], rob_dict['home_pos'][0]['y'], rob_dict['home_pos'][0]['z']])
    stateMachine = RobotStateMachine(control_pipe, gripper_pose_estimator, encoder_pos_m, 
                                     home_xyz_coords, constants = PP_CONSTS, verbose = True)


    if DETECTOR_TYPE == 'deep_1':
        show_boot_screen('STARTING NEURAL NET...')
        pack_detect = PacketDetector(paths, files, check_point, 3)
    elif DETECTOR_TYPE == 'deep_2':
        # TODO Implement new deep detector
        pass
    elif DETECTOR_TYPE == 'hsv':
        pack_detect = ThresholdDetector(ignore_vertical_px = 133, ignore_horizontal_px = 50, max_ratio_error = 0.15,
                                        white_lower = [60, 0, 85], white_upper = [179, 255, 255],
                                        brown_lower = [0, 33, 57], brown_upper = [60, 255, 178])

    # Toggles
    toggles_dict = {
        'gripper' : False, # Gripper state
        'conv_left' : False, # Conveyor heading left enable
        'conv_right' : False, # Conveyor heading right enable
        'show_bbox' : True, # Bounding box visualization enable
        'show_frame_data' : False, # Show frame data (robot pos, encoder vel, FPS ...)
        'show_depth_map' : False, # Overlay colorized depth enable
        'show_hsv_mask' : False, # Remove pixels not within HSV mask boundaries
    }

    # Variables
    frame_count = 1 # Counter of frames for homography update
    text_size = 1
    homography = None # Homography matrix
    

    # Set home position from dictionary on startup
    control_pipe.send(RcData(RcCommand.SET_HOME_POS_SH))

    while True:
        # Start timer for FPS estimation
        start_time = time.time()

        # READ DATA
        ###################

        # Read data dict from PLC server
        try:
            rob_stopped = info_dict['rob_stopped']
            stop_active = info_dict['stop_active']
            prog_done = info_dict['prog_done']
            encoder_vel = info_dict['encoder_vel']
            pos = info_dict['pos']
            speed_override = info_dict['speed_override']
        except:
            continue

        # Read encoder dict from PLC server
        encoder_pos = encoder_pos_m.value
        if encoder_pos is None:
            continue

        # Get frames from realsense
        success, depth_frame, rgb_frame, colorized_depth = dc.get_aligned_frame()
        if not success:
            continue

        frame_height, frame_width, frame_channel_count = rgb_frame.shape
        text_size = (frame_height / 1000)
        image_frame = rgb_frame.copy()

        # Draw HSV mask over screen if enabled
        if toggles_dict['show_hsv_mask'] and DETECTOR_TYPE == 'hsv':
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
            if frame_count >= MAX_FRAME_COUNT:
                frame_count = 1

            # Set homography in HSV detector
            if DETECTOR_TYPE == 'hsv':
                pack_detect.set_homography(homography)

        # PACKET DETECTION
        ##################
        
        # Detect packets using neural network
        if DETECTOR_TYPE == 'deep_1':
            image_frame, detected_packets = pack_detect.deep_pack_obj_detector(rgb_frame, 
                                                                               depth_frame,
                                                                               encoder_pos,
                                                                               bnd_box = toggles_dict['show_bbox'],
                                                                               homography = homography,
                                                                               image_frame = image_frame)
            for packet in detected_packets:
                packet.width = packet.width * frame_width
                packet.height = packet.height * frame_height
        
        # Detect packets using neural network
        elif DETECTOR_TYPE == 'deep_2':
            # TODO Implement new deep detector
            detected_packets = []
            pass
        
        # Detect packets using neural HSV thresholding
        elif DETECTOR_TYPE == 'hsv':
            image_frame, detected_packets, mask = pack_detect.detect_packet_hsv(rgb_frame,
                                                                          encoder_pos,
                                                                          draw_box = toggles_dict['show_bbox'],
                                                                          image_frame = image_frame)

        registered_packets = packet_tracking(pt, detected_packets, depth_frame, frame_width, mask)

        # STATE MACHINE
        ###############
        # Robot ready when programs are fully finished and it isn't moving
        is_rob_ready = prog_done and (rob_stopped or not stop_active)  
        stateMachine.run(homography, is_rob_ready, registered_packets, encoder_vel, pos)

        # FRAME GRAPHICS
        ################
        draw_frame(image_frame, registered_packets, encoder_pos, text_size, toggles_dict, info_dict, colorized_depth, start_time, frame_width, frame_height)
        
        # Keyboard inputs
        key = cv2.waitKey(1)
        end_prog, toggles_dict = process_key_input(key, control_pipe, toggles_dict, is_rob_ready)
        
        # End main
        if end_prog:
            control_pipe.send(RcData(RcCommand.CLOSE_PROGRAM))
            cv2.destroyAllWindows()
            dc.release()
            break