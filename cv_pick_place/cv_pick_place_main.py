import os
import cv2 
import json
import time
import numpy as np
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Pipe

from robot_cell.packet.packet_object import Packet
from robot_cell.packet.item_object import Item
from robot_cell.packet.packettracker import PacketTracker
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.packet.point_cloud_viz import PointCloudViz
from robot_cell.packet.centroidtracker import CentroidTracker
from robot_cell.control.robot_communication import RobotCommunication
from robot_cell.control.robot_control import RobotControl
from robot_cell.control.robot_control import RcCommand
from robot_cell.control.robot_control import RcData
from robot_cell.control.pick_place_demos import RobotDemos
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
FRAMES_LIM = 10 # Max frames object must be tracked to start pick & place
PACK_DEPTHS = [10.0, 3.0, 5.0, 5.0] # Predefined packet depths, index corresponds to type of packet.
MIN_PICK_DISTANCE = 600  # Minimal x position in mm for packet picking
MAX_PICK_DISTANCE = 1900 # Maximal x position in mm for packet picking
Z_OFFSET = 50.0 # Z height offset from pick height for all positions except for pick position
X_PICK_OFFSET = 140 # X offset between prepick and pick position
GRIP_TIME_OFFSET = 400  # X offset from current packet position to prepick position
PICK_START_X_OFFSET = 25 # Offset between robot and packet for starting the pick move
# TODO Vit please add comments
MAX_Z = 500
MIN_Y = 45.0
MAX_Y = 470.0

def main(rob_dict, paths, files, check_point, info_dict, encoder_pos_m, control_pipe):
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
    home_xyz_coords = np.array([rob_dict['home_pos'][0]['x'],
                                rob_dict['home_pos'][0]['y'],
                                rob_dict['home_pos'][0]['z']])
    stateMachine = RobotStateMachine(control_pipe, gripper_pose_estimator, encoder_pos_m, home_xyz_coords, verbose = True)


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
        'gripper' : False,
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


#------------------------------FUNCTIONS----------------------------------------------------------------
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
            if item.centroid[0] - item.width / 2 > item.crop_border_px and item.centroid[0] + item.width / 2 < (frame_width - item.crop_border_px):
                depth_crop = item.get_crop_from_frame(depth_frame)
                mask_crop = item.get_crop_from_frame(mask)
                item.add_depth_crop_to_average(depth_crop)
                item.set_mask(mask_crop)

    # Update registered packet list with new packet info
    registered_packets = pt.item_database

    return registered_packets

class RobotStateMachine:
    """ State machine for robot control
    """
    def __init__(self, control_pipe, gripper_pose_estimator, encoder_pos_m, home_xyz_coords, verbose = False):
            # input
            self.cp = control_pipe
            self.gpe = gripper_pose_estimator
            self.enc_pos = encoder_pos_m
            self.home_xyz_coords = home_xyz_coords
            self.verbose = verbose
            # init
            self.state = 'READY'
            self.pick_list = []
            self.is_in_home_pos = False
            self.prepick_xyz_coords = []
            self.packet_to_pick = None
            self.trajectory_dict = {}
        # TODO add comments to variables
        
    def run(self, homography, is_rob_ready, registered_packets, encoder_vel, pos):
        """Run the state machine

        Args:
            homography (numpy.ndarray): Homography matrix.
            is_rob_ready (bool): indication if robot is ready to start
            registered_packets (list[packets]): _description_
            encoder_vel (double): encoder velocity
            pos (np.ndarray): current robot position
        Returns:
            state (string): current state
        """
        
        self.pick_list = add_to_pick_list(self.pick_list, registered_packets, encoder_vel) 

        # robot is ready to recieve commands
        if self.state == "READY" and is_rob_ready and homography is not None:
            self.pick_list, pick_list_positions = prep_pick_list(self.pick_list, self.enc_pos, homography)
            # Choose a item for picking
            if self.pick_list and pick_list_positions.max() > MIN_PICK_DISTANCE:
                self.packet_to_pick, self.trajectory_dict = start_program(self.pick_list, pick_list_positions, homography, self.gpe, self.cp)
                # Save prepick position for use in TO_PREPICK state
                self.prepick_xyz_coords = np.array([self.trajectory_dict['x'], self.trajectory_dict['y'], self.trajectory_dict['pack_z'] + Z_OFFSET])
                self.is_in_home_pos = False
                self.state = "TO_PREPICK"
                if self.verbose : print("[INFO]: state TO_PREPICK")
            # send robot to home position if it itsn't already
            elif not self.is_in_home_pos:
                self.cp.send(RcData(RcCommand.GO_TO_HOME))
                self.state = "TO_HOME_POS"
                if self.verbose : print("[INFO]: state TO_HOME_POS")

        # moving to home position
        if self.state == "TO_HOME_POS":
            if is_rob_ready and is_rob_in_pos(pos, self.home_xyz_coords):
                self.is_in_home_pos = True
                self.state = "READY"
                if self.verbose : print("[INFO]: state READY")

        # moving to prepick position
        if self.state == "TO_PREPICK":
            # check if robot arrived to prepick position
            if is_rob_in_pos(pos, self.prepick_xyz_coords):
                self.state = "WAIT_FOR_PACKET"
                if self.verbose : print("[INFO]: state WAIT_FOR_PACKET")

        # waiting for packet
        if self.state == "WAIT_FOR_PACKET":
            encoder_pos = self.enc_pos.value
            # check encoder and activate robot 
            self.packet_to_pick.centroid = self.packet_to_pick.getCentroidFromEncoder(encoder_pos)
            packet_pos_x = self.packet_to_pick.getCentroidInWorldFrame(homography)[0]
            # If packet is too far abort and return to ready
            if packet_pos_x > self.trajectory_dict['x'] + X_PICK_OFFSET:
                self.cp.send(RcData(RcCommand.CONTINUE_PROGRAM))
                self.cp.send(RcData(RcCommand.ABORT_PROGRAM))
                self.cp.send(RcData(RcCommand.GRIPPER, False))
                self.state = "READY"
                if self.verbose : print("[INFO]: missed packet, state READY")

            # If packet is close enough continue picking operation
            elif packet_pos_x > self.trajectory_dict['x'] - PICK_START_X_OFFSET - self.trajectory_dict['shift_x']:
                self.cp.send(RcData(RcCommand.CONTINUE_PROGRAM))
                self.state = "PLACING"
                if self.verbose : print("[INFO]: state PLACING")

        # placing packet
        if self.state == "PLACING":
            if is_rob_ready:
                self.state = "READY"
                if self.verbose : print("[INFO]: state READY")

        return self.state



def add_to_pick_list(pick_list, registered_packets, encoder_vel):
    """Add packets which have been tracked for FRAMES_LIM frames to the pick list

    Args:
        pick_list (list[packets]): list of packets ready to be picked
        registered_packets (list[packet_object]): List of tracked packet objects
        encoder_vel (double): encoder velocity
    """
    # When speed of conveyor more than -100 it is moving to the left
    is_conv_mov = encoder_vel < - 100.0

    # Add to pick list
    # If packets are being tracked.
    if registered_packets:
        # If the conveyor is moving to the left direction.
        if is_conv_mov:
            # Increase counter of frames with detections.
            for packet in registered_packets:
                packet.track_frame += 1
                # If counter larger than limit, and packet not already in pick list.
                if packet.track_frame > FRAMES_LIM and not packet.in_pick_list:
                    print("[INFO]: Add packet ID: {} to pick list".format(str(packet.id)))
                    # Add to pick list.
                    packet.in_pick_list = True
                    pick_list.append(packet)

    return pick_list


def prep_pick_list(pick_list, encoder_pos_m, homography):
    """
    Prepare the list for choosing a packet by updating packet positions
    and removing packets which are too far.

    Args:
        pick_list (list): List of packets ready to be picked, contains packet type objects
        encoder_pos_m (multiprocessing.value): Value object from multiprocessing Manager for reading encoder value from another process
        homography (numpy.ndarray): Homography matrix
        GRIP_TIME_OFFSET (int): x position offset in mm for prepick position
        X_PICK_OFFSET (int): X offset between prepick and pick position
        MAX_PICK_DISTANCE (int): Maximal x position in mm for packet picking

    Returns:
        pick_list: (list): Updated pick list
        pick_list_positions (list[int]): List of current position of packets
    """
    encoder_pos = encoder_pos_m.value
    if encoder_pos is None:
        return [], []
    # Update pick list to current positions
    for packet in pick_list:
        packet.centroid = packet.getCentroidFromEncoder(encoder_pos)
    # Get list of current world x coordinates
    pick_list_positions = np.array([packet.getCentroidInWorldFrame(homography)[0] for packet in pick_list])
    # If item is too far remove it from list
    is_valid_position = pick_list_positions < MAX_PICK_DISTANCE - GRIP_TIME_OFFSET - 1.5*X_PICK_OFFSET
    pick_list = np.ndarray.tolist(np.asanyarray(pick_list)[is_valid_position])     
    pick_list_positions = pick_list_positions[is_valid_position]
    return pick_list, pick_list_positions


def get_pick_positions(packet_to_pick, homography, gripper_pose_estimator):
    """
    Get dictionary of parameters for changing trajectory

    Args:
        packet_to_pick (packet): Packet choosen for picking
        homography (numpy.ndarray): Homography matrix.
        gripper_pose_estimator: TODO
    Returns:
        trajectory_dict (dict): Dictionary of parameters for changing trajectory 
    """
    # TODO finish doc string
    # Set positions and Start robot
    packet_x,pick_pos_y = packet_to_pick.getCentroidInWorldFrame(homography)
    pick_pos_x = packet_x + GRIP_TIME_OFFSET

    # offs = get_gripper_offset(np.array([packet_x, pick_pos_y]), np.array(pos[0:2]))
    # pick_pos_x = packet_x + offs

    angle = packet_to_pick.angle
    gripper_rot = compute_gripper_rot(angle)        # TODO: Use rotation
    packet_type = packet_to_pick.type

    # Set packet depth to fixed value by type                
    # Prediction of position by the gripper pose estimation
    # Limiting the height for packet pick positions
    z_lims = (PACK_DEPTHS[packet_to_pick.type], MAX_Z)
    packet_coords = (pick_pos_x, pick_pos_y)
    y_lims = (MIN_Y, MAX_Y)
    shift_x, shift_y, pick_pos_z, roll, pitch, yaw = gripper_pose_estimator.estimate_from_packet(packet_to_pick, z_lims, y_lims, packet_coords)
    if shift_x is not None:
        print(f"[INFO]: Estimated optimal point:\n\tx, y shifts: {shift_x:.2f}, {shift_y:.2f},\
                \n\tz position: {pick_pos_z:.2f}\n\tRPY angles: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
        pick_pos_y += shift_y
    else: 
        # TODO: Implement behaviour in the future 
        # Either continue with centroid or skip packet IDK, TBD
        pass

    # Check if x is range
    pick_pos_x = np.clip(pick_pos_x, MIN_PICK_DISTANCE, MAX_PICK_DISTANCE - 1.5*X_PICK_OFFSET)
    # Check if y is range of conveyor width and adjust accordingly
    pick_pos_y = np.clip(pick_pos_y, 75.0, 470.0)
    # Offset pick height by position on belt
    pick_pos_z = offset_packet_depth_by_x(pick_pos_x, pick_pos_z)
    

    # Change end points of robot.   
    trajectory_dict = {
        'x': pick_pos_x,
        'y': pick_pos_y,
        'rot': gripper_rot,
        'packet_type': packet_type,
        'x_offset': X_PICK_OFFSET,
        'pack_z': pick_pos_z,
        'a': roll,
        'b': pitch,
        'c': yaw,
        'z_offset': Z_OFFSET,
        'shift_x': shift_x
        }

    return trajectory_dict


def start_program(pick_list, pick_list_positions, homography, gripper_pose_estimator, control_pipe):
    """
    Choose a packet from pick list and start programm.
    
    Args:
        pick_list (list[packet]): list of items ready to be picked_
        pick_list_positions (numpy.ndarray[float64]): List of current x positions for items in pick list
        homography (numpy.ndarray): Homography matrix.
        gripper_pose_estimator (class): Class for estimating gripper angles
        control_pipe (Pipe): Communication pipe between processes
    Returns:
        packet_to_pick (packet): Packet choosen for picking
        trajectory_dict (dict): Dictionary of parameters for changing trajectory 
    """
    # Chose farthest item on belt
    pick_ID = pick_list_positions.argmax()
    packet_to_pick = pick_list.pop(pick_ID)
    print("[INFO]: Chose packet ID: {} to pick".format(str(packet_to_pick.id)))

    trajectory_dict = get_pick_positions(packet_to_pick, homography, gripper_pose_estimator)


    # Set trajectory
    control_pipe.send(RcData(RcCommand.CHANGE_SHORT_TRAJECTORY, trajectory_dict))
    # Start robot program.
    control_pipe.send(RcData(RcCommand.START_PROGRAM, True))

    return packet_to_pick, trajectory_dict


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
    for packet in registered_packets:
        if packet.avg_depth_crop is not None:
            cv2.imshow("Depth Crop", colorizeDepthFrame(packet.avg_depth_crop))
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







#------------------------- MODE SELECT --------------------------------------------------
def program_mode(demos, r_control, r_comm_info, r_comm_encoder):
    """
    Program selection function.

    Parameters:
    rc (object): RobotControl object for program execution.
    rd (object): RobotDemos object containing pick and place demos.

    """
    # Read mode input.
    mode = input()
    # Dictionary with robot positions and robot programs.
    modes_dict = {
        '1':{'dict':Pick_place_dict, 
            'func':demos.main_robot_control_demo},
        '2':{'dict':Pick_place_dict, 
            'func':demos.main_pick_place},
        '3':{'dict':Pick_place_dict_conv_mov, 
            'func':demos.main_pick_place_conveyor_w_point_cloud},
        '4':{'dict':Short_pick_place_dict, 
            'func':main}
                }

    # If mode is a program key.
    if mode in modes_dict:
        # Set robot positions and robot program
        r_control.rob_dict = modes_dict[mode]['dict']
        robot_prog = modes_dict[mode]['func']

        # If first mode (not threaded) start program
        if mode == '1':
            robot_prog(r_control)

        elif mode == '4':
            with Manager() as manager:
                info_dict = manager.dict()
                encoder_pos = manager.Value('d', None)

                control_pipe_1, control_pipe_2 = Pipe()

                main_proc = Process(target = robot_prog, args = (r_control.rob_dict, paths, files, check_point, info_dict, encoder_pos, control_pipe_1))
                info_server_proc = Process(target = r_comm_info.robot_server, args = (info_dict, ))
                encoder_server_proc = Process(target = r_comm_encoder.encoder_server, args = (encoder_pos, ))
                control_server_proc = Process(target = r_control.control_server, args = (control_pipe_2, ))

                main_proc.start()
                info_server_proc.start()
                encoder_server_proc.start()
                control_server_proc.start()

                # Wait for the main process to end
                main_proc.join()
                info_server_proc.kill()
                encoder_server_proc.kill()
                control_server_proc.kill()

        # Otherwise start selected threaded program
        else:
            with Manager() as manager:
                info_dict = manager.dict()

                main_proc = Process(target = robot_prog, args = (r_control, paths, files, check_point, info_dict))
                info_server_proc = Process(target = r_comm_info.robot_server, args = (info_dict, ))

                main_proc.start()
                info_server_proc.start()

                # Wait for the main process to end
                main_proc.join()
                info_server_proc.kill()

    # If input is exit, exit python.
    if mode == 'e':
        exit()

    # Return function recursively.
    return program_mode(demos, r_control, r_comm_info, r_comm_encoder)


if __name__ == '__main__':
    # Define model parameters.
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    check_point ='ckpt-3'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    # Define model paths.
    paths = {'ANNOTATION_PATH':os.path.join(
                                            'Tensorflow',
                                            'workspace',
                                            'annotations'),
            'CHECKPOINT_PATH': os.path.join(
                                            'Tensorflow', 
                                            'workspace',
                                            'models',
                                            CUSTOM_MODEL_NAME) 
            }
    files = {'PIPELINE_CONFIG':os.path.join(
                                            'Tensorflow', 
                                            'workspace',
                                            'models',
                                            CUSTOM_MODEL_NAME,
                                            'pipeline.config'),
            'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], 
                                        LABEL_MAP_NAME)
            }

    # Define robot positions dictionaries from json file.
    file = open('robot_positions.json')
    robot_poses = json.load(file)
    Pick_place_dict_conv_mov_slow = robot_poses['Pick_place_dict_conv_mov_slow']
    Pick_place_dict_conv_mov = robot_poses['Pick_place_dict_conv_mov']
    Pick_place_dict = robot_poses['Pick_place_dict']
    Short_pick_place_dict = robot_poses['Short_pick_place_dict']
    
    # Initialize robot demos and robot control objects.
    r_control = RobotControl(None)
    r_comm_info = RobotCommunication()
    r_comm_encoder = RobotCommunication()
    demos = RobotDemos(paths, files, check_point)

    # Show message about robot programs.
    print('Select pick and place mode: \n'+
    '1 : Pick and place with static conveyor and hand gestures\n'+
    '2 : Pick and place with static conveyor and multithreading\n'+
    '3 : Pick and place with moving conveyor, point cloud and multithreading\n'+
    '4 : Main Pick and place\n'+
    'e : To exit program\n')

    # Start program mode selection.
    program_mode(demos, r_control, r_comm_info, r_comm_encoder)