import os
import sys
import cv2 
import json
import time
import random
import datetime
import numpy as np
import pyrealsense2
import scipy.signal
from opcua import ua
from opcua import Client
import matplotlib as mpl
from scipy import ndimage
from queue import Queue
from threading import Thread
from threading import Timer
from collections import OrderedDict
from scipy.spatial import distance as dist
from multiprocessing import Process
from multiprocessing import Pipe

from robot_cell.packet.packet_object import Packet
from robot_cell.packet.item_object import Item
from robot_cell.packet.packettracker import PacketTracker
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.packet.point_cloud_viz import PointCloudViz
from robot_cell.packet.centroidtracker import CentroidTracker
from robot_cell.control.robot_communication import RobotCommunication
from robot_cell.control.robot_control import RobotControl
from robot_cell.control.pick_place_demos import RobotDemos
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag

from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.functions import *

USE_DEEP_DETECTOR = False

def main(rc, paths, files, check_point, connection_info, connection_encoder):
    """
    Thread for pick and place with moving conveyor and point cloud operations.
    
    Parameters:
    rc (object): RobotControl object for program execution.
    server_in (object): Queue object containing data from the PLC server.
    
    """
    # Inititalize objects.
    apriltag = ProcessingApriltag()
    pt = ItemTracker(max_disappeared_frames = 20, guard = 100, max_item_distance=500)
    dc = DepthCamera()

    if USE_DEEP_DETECTOR:
        rc.show_boot_screen('STARTING NEURAL NET...')
        pack_detect = PacketDetector(paths, files, check_point)
    else:
        pack_detect = ThresholdDetector()

    # Initialize list for items ready to be picked
    pick_list = []
    
    robot_server_dict = None
    encoder_pos = None
    rc.connect_OPCUA_server()
    rc.get_nodes()

    # Define fixed x position where robot waits for packet.
    x_fixed = rc.rob_dict['pick_pos_base'][0]['x']

    # Declare variables.
    warn_count = 0 # Counter for markers out of frame or moving.
    track_frame = 0 # Counter for num of frames when object is tracked.
    frames_lim = 10 # Max frames object must be tracked to start pick&place.
    frame_count = 1 # Counter of frames for homography update.
    bbox = True # Bounding box visualization enable.
    f_data = False # Show frame data (robot pos, encoder vel, FPS ...).
    depth_map = False # Overlay colorized depth enable.
    is_detect = False # Detecting objects enable.
    conv_left = False # Conveyor heading left enable.
    conv_right = False # Conveyor heading right enable.
    homography = None # Homography matrix.
    track_result = None # Result of pack_obj_tracking_update.
    text_size = 1

    # robot state variable
    state = "READY"

    # Predefine packet z and x offsets with robot speed of 55%.
    # Index corresponds to type of packet.
    pack_depths = [10.0, 3.0, 5.0, 5.0] # List of z positions at pick.
    pack_x_offsets = [50.0, 180.0, 130.0, 130.0] # List of x positions at pick.

    while True:
        # Start timer for FPS estimation
        start_time = time.time()

        # Read data dict from PLC server
        if connection_info.poll():
            robot_server_dict = connection_info.recv()
            rob_stopped = robot_server_dict['rob_stopped']
            stop_active = robot_server_dict['stop_active']
            prog_done = robot_server_dict['prog_done']
            encoder_vel = robot_server_dict['encoder_vel']
            pos = robot_server_dict['pos']
        elif robot_server_dict is None:
            continue

        # Read encoder value from PLC server
        if connection_encoder.poll():
            encoder_pos = connection_encoder.recv()
        elif encoder_pos is None:
            continue

        # Get frames from realsense.
        success, depth_frame, rgb_frame, colorized_depth = dc.get_aligned_frame()
        if not success:
            continue

        # Crop and resize depth frame to match rgb frame.
        frame_height, frame_width, frame_channel_count = rgb_frame.shape

        text_size = (frame_height / 1000)

        try:
            # Try to detect tags in rgb frame.
            rgb_frame = apriltag.detect_tags(rgb_frame)

            # Update homography on first frame of 500 frames.
            if frame_count == 1:
                homography = apriltag.compute_homog()
                print('[INFO]: Homography matrix updated.')

            # If recieving homography matrix as np array.
            is_type_np = type(homography).__module__ == np.__name__
            is_marker_detect = is_type_np or homography == None

            if not USE_DEEP_DETECTOR:
                pack_detect.set_homography(homography)

            # Reset not detected tags warning.
            if is_marker_detect:
                warn_count = 0
                
        # Triggered when no markers are in the frame.
        except Exception as e:
            warn_count += 1
            # Print warning only once.
            if warn_count == 1:
                print(e)
                print("[INFO]: Markers out of frame or moving.")
            pass
        
        # Detect packets using neural network.
        if USE_DEEP_DETECTOR:
            image_frame, detected_packets = pack_detect.deep_pack_obj_detector(rgb_frame, 
                                                                               depth_frame,
                                                                               encoder_pos,
                                                                               bnd_box=bbox)
        else:
            image_frame, detected_packets = pack_detect.detect_packet_hsv(rgb_frame,
                                                                          depth_frame,
                                                                          encoder_pos,
                                                                          bbox,
                                                                          text_size)

        # Update tracked packets for current frame.
        labeled_packets = pt.track_items(detected_packets)
        pt.update_item_database(labeled_packets)
        registered_packets = pt.item_database
        # print({
        #     'packs': registered_packets,
        #     'rob_stop': rob_stopped,
        #     'stop_acti': stop_active,
        #     'prog_done': prog_done})

        # When detected not empty, objects are being detected.
        is_detect = len(detected_packets) != 0
        # When speed of conveyor more than -100 it is moving to the left.
        is_conv_mov = encoder_vel < - 100.0
        #Robot ready when programs are fully finished and it isn't moving.
        is_rob_ready = prog_done and (rob_stopped or not stop_active)


        # TODO remove VIS TEST 
        for packet in registered_packets:
            if packet.disappeared == 0:
                # Draw packet ID
                text_id = "ID {}".format(packet.id)
                drawText(image_frame, text_id, (packet.centroid[0] + 10, packet.centroid[1]), text_size)

                # Draw packet centroid
                text_centroid = "X: {}, Y: {}".format(packet.centroid[0], packet.centroid[1])
                drawText(image_frame, text_centroid, (packet.centroid[0] + 10, packet.centroid[1] + int(45 * text_size)), text_size)
                
                # print("packet ID: {}, tracked: {}, ".format(str(packet.id), str(packet.track_frame)))

        # Add to pick list
        # If packets are being tracked.
        if is_detect:
            # If the conveyor is moving to the left direction.
            if is_conv_mov:
                # Increase counter of frames with detections. 
                for packet in registered_packets:
                    packet.track_frame += 1
                    # If counter larger than limit, and packet not already in pick list.
                    if packet.track_frame > frames_lim and not packet.in_pick_list:
                        print("INFO: Add packet ID: {} to pick list".format(str(packet.id)))
                        # Add to pick list.
                        packet.in_pick_list = True
                        pick_list.append(packet) # ? copy

        # ROBOT READY 
        if state == "READY" and is_rob_ready and pick_list and homography is not None:
            # Update pick list to current positions
            for packet in pick_list:
                packet.centroid = packet.getCentroidFromEncoder(encoder_pos)
            # Get list of current world x coordinates
            pick_list_positions = np.array([packet.getCentroidInWorldFrame(homography)[0] for packet in pick_list])
            print("DEBUG: Pick distances")
            print(pick_list_positions)
            # If item is too far remove it from list
            is_valid_position = pick_list_positions < 1150   # TODO find position after which it does not pick up - depends on enc_vel and robot speed
            pick_list = np.ndarray.tolist(np.asanyarray(pick_list)[is_valid_position])     
            pick_list_positions = pick_list_positions[is_valid_position]
            print("DEBUG: Pick distances after removal")
            print(pick_list_positions)
            # Choose a item for picking
            if pick_list:
                # Chose farthest item on belt
                pick_ID = pick_list_positions.argmax()
                packet_to_pick = pick_list.pop(pick_ID)
                print("INFO: Chose packet ID: {} to pick".format(str(packet.id)))

                # Set positions and Start robot
                packet_x,packet_y = packet_to_pick.getCentroidInWorldFrame(homography)
                packet_x = x_fixed  # for testing # TODO find offset value from packet
                angle = packet.angle
                gripper_rot = rc.compute_gripper_rot(angle)
                packet_type = packet.pack_type

                # Set packet depth to fixed value by type
                packet_z = pack_depths[packet_type]

                # Check if y is range of conveyor width and adjust accordingly.
                if packet_y < 75.0:
                    packet_y = 75.0

                elif packet_y > 470.0:
                    packet_y = 470.0
                # TODO clamp x position when it's variable

                prepick_xyz_coords = np.array([packet_x, packet_y, rc.rob_dict['pick_pos_base'][0]['z']])

                # Change end points of robot.   
                rc.change_trajectory(
                                packet_x,
                                packet_y, 
                                gripper_rot, 
                                packet_type,
                                x_offset = pack_x_offsets[packet_type],
                                pack_z = packet_z)

                # Start robot program.   #! only once
                rc.start_program()
                state = "TO_PREPICK"
                print("state: TO_PREPICK")


        # TO PREPICK
        if state == "TO_PREPICK":
            # check if robot arrived to prepick position
            curr_xyz_coords = np.array(pos[0:3])
            robot_dist = np.linalg.norm(prepick_xyz_coords-curr_xyz_coords)
            if robot_dist > 10: # TODO check value
                state = "WAIT_FOR_PACKET"
                print("state: WAIT_FOR_PACKET")

        # WAIT FOR PACKET
        if state == "WAIT_FOR_PACKET":
            # TODO add return to ready if it misses packet
            # check encoder and activate robot 
            packet_to_pick.centroid = packet_to_pick.getCentroidFromEncoder(encoder_pos)
            p_x = packet_to_pick.getCentroidInWorldFrame(homography)[0]
            print("X distance")
            print(packet_x - p_x)
            if p_x > packet_x -70:
                # print("DEBUG: Fast encoder update")
                # while  p_x < packet_x - 20:
                #     encoder_pos = round(rc.Encoder_Pos.get_value(),2)
                #     packet_to_pick.centroid = packet_to_pick.getCentroidFromEncoder(encoder_pos)
                #     p_x = packet_to_pick.getCentroidInWorldFrame(homography)[0]
                #     print("X distance")
                #     print(packet_x - p_x)
                rc.continue_program()
                state = "PICKING"
                print("state: PICKING")

        if state == "PICKING":
            if is_rob_ready:
                state = "READY"
                print("state: READY")

        # Show point cloud visualization when packets are deregistered.
        # if len(deregistered_packets) > 0:
        #     pclv = PointCloudViz(".", deregistered_packets[-1])
        #     pclv.show_point_cloud()
        #     del pclv

        # Show depth frame overlay.
        if depth_map:
            image_frame = cv2.addWeighted(image_frame, 0.8, colorized_depth, 0.3, 0)

        # Show FPS and robot position data
        if f_data:
            # Draw FPS to screen
            text_fps = "FPS: {:.2f}".format(1.0 / (time.time() - start_time))
            drawText(image_frame, text_fps, (10, int(35 * text_size)), text_size)

            # Draw OPCUA data to screen
            text_robot = str(robot_server_dict)
            drawText(image_frame, text_robot, (10, int(75 * text_size)), text_size)

        # Show frames on cv2 window
        image_frame = cv2.resize(image_frame, (frame_width // 2, frame_height // 2))
        cv2.imshow("Frame", image_frame)

        # Increase counter for homography update.
        frame_count += 1
        if frame_count == 500:
            frame_count = 1
        
        # Keyboard inputs.
        key = cv2.waitKey(1)

        if key == ord('o'):
            rc.change_gripper_state(False)

        if key == ord('i'):
            rc.change_gripper_state(True)

        if key == ord('m'):
            conv_right = rc.change_conveyor_right(conv_right)
        
        if key == ord('n'):
            conv_left = rc.change_conveyor_left(conv_left)

        if key == ord('l'):
            bbox = not bbox
        
        if key == ord('h'):
            depth_map = not depth_map
                
        if key == ord('f'):
            f_data = not f_data
        
        if key == ord('e'):
            is_detect = not is_detect

        if key == ord('a'):
            rc.abort_program()
        
        if key == ord('c'):
            rc.continue_program()
        
        if key == ord('s'):
            rc.stop_program()

        if key == ord('r'):
            print(is_rob_ready)
        
        if key == 27:
            rc.close_program()
            break

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
        '4':{'dict':Pick_place_dict_conv_mov, 
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

        # Otherwise start selected threaded program
        else:
            pipe_1_in, pipe_1_out = Pipe()
            pipe_2_in, pipe_2_out = Pipe()
            main_proc = Process(target = robot_prog, args = (r_control, paths, files, check_point, pipe_1_out, pipe_2_out))
            info_server_proc = Process(target = r_comm_info.robot_server, args = (pipe_1_in, ))
            encoder_server_proc = Process(target = r_comm_encoder.encoder_server, args = (pipe_2_in, ))

            main_proc.start()
            info_server_proc.start()
            encoder_server_proc.start()

            # Wait for the main process to end
            main_proc.join()
            info_server_proc.kill()
            encoder_server_proc.kill()

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
