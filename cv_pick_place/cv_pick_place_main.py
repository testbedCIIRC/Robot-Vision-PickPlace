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

from robot_cell.packet.packet_object import Packet
from robot_cell.packet.item_object import Item
from robot_cell.packet.packettracker import PacketTracker
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.packet.point_cloud_viz import PointCloudViz
from robot_cell.packet.centroidtracker import CentroidTracker
from robot_cell.control.robot_control import RobotControl
from robot_cell.control.pick_place_demos import RobotDemos
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag

from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.control.fake_robot_control import FakeRobotControl

def main(rc, server_in):
    """
    Thread for pick and place with moving conveyor and point cloud operations.
    
    Parameters:
    rc (object): RobotControl object for program execution.
    server_in (object): Queue object containing data from the PLC server.
    
    """
    # Inititalize objects.
    apriltag = ProcessingApriltag()
    # pt = ItemTracker(maxDisappeared=10, guard=30, max_item_distance=300)    
    pt = PacketTracker(maxDisappeared=10, guard=30)    
    dc = DepthCamera()
    rc.show_boot_screen('STARTING NEURAL NET...')
    pack_detect = ThresholdDetector()

    # Initialize list for items ready to be picked
    pick_list = []

    # Define fixed x position where robot waits for packet.
    x_fixed = rc.rob_dict['pick_pos_base'][0]['x']

    # Declare variables.
    warn_count = 0 # Counter for markers out of frame or moving.
    track_frame = 0 # Counter for num of frames when object is tracked.
    frames_lim = 10 # Max frames object must be tracked to start pick&place.
    frame_count = 1 # Counter of frames for homography update.
    bbox = True # Bounding box visualization enable.
    f_data = False # Show frame data (robot pos, encoder vel, FPS ...).
    depth_map = True # Overlay colorized depth enable.
    is_detect = False # Detecting objects enable.
    conv_left = False # Conveyor heading left enable.
    conv_right = False # Conveyor heading right enable.
    homography = None # Homography matrix.
    track_result = None # Result of pack_obj_tracking_update.

    # robot state variable
    state = "READY"

    # Predefine packet z and x offsets with robot speed of 55%.
    # Index corresponds to type of packet.
    pack_depths = [10.0, 3.0, 5.0, 5.0] # List of z positions at pick.
    pack_x_offsets = [50.0, 180.0, 130.0, 130.0] # List of x positions at pick.
    
    while True:
        # Start timer for FPS estimation.
        start_time = time.time()

        # Read data dict from PLC server stored in queue object.
        robot_server_dict = server_in.get()
        rob_stopped = robot_server_dict['rob_stopped']
        stop_active = robot_server_dict['stop_active']
        prog_done = robot_server_dict['prog_done']
        encoder_vel = robot_server_dict['encoder_vel']
        encoder_pos = robot_server_dict['encoder_pos']
        pos = robot_server_dict['pos']

        # Get frames from realsense.
        success, depth_frame, rgb_frame, colorized_depth = dc.get_aligned_frame()
        if not success:
            pass

        # Crop and resize depth frame to match rgb frame.
        height, width, depth = rgb_frame.shape

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
        img_detect, detected = pack_detect.detect_packet_hsv(rgb_frame, 
                                                             depth_frame,
                                                             encoder_pos)
        # Update tracked packets for current frame.
        registered_packets, deregistered_packets = pt.update(detected, depth_frame)
        # print({
        #     'packs': registered_packets,
        #     'rob_stop': rob_stopped, 
        #     'stop_acti': stop_active, 
        #     'prog_done': prog_done})

        # When detected not empty, objects are being detected.
        is_detect = len(detected) != 0 
        # When speed of conveyor more than -100 it is moving to the left.
        is_conv_mov = encoder_vel < - 100.0
        #Robot ready when programs are fully finished and it isn't moving.
        is_rob_ready = prog_done and (rob_stopped or not stop_active)


        # TODO remove VIS TEST 
        for (objectID, packet) in registered_packets.items():
            # Draw both the ID and centroid of packet objects.
            centroid_tup = packet.centroid
            centroid = np.array([centroid_tup[0],centroid_tup[1]]).astype('int')
            text = "ID {}".format(objectID)
            cv2.putText(img_detect, text, (centroid[0] , centroid[1] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(img_detect, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)
            cv2.circle(img_detect, packet.getCentroidFromEncoder(encoder_pos), 4, (0, 0, 255), -1)
        #     print("packet ID: {}, tracked: {}, ".format(str(packet.id), str(packet.track_frame)))
        # print("PICK LIST")
        # print(pick_list)

        # Add to pick list
        # If packets are being tracked.
        if is_detect:
            # If the conveyor is moving to the left direction.
            if is_conv_mov:
                # Increase counter of frames with detections. 
                for (objectID, packet) in registered_packets.items():
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
            is_valid_position = pick_list_positions < 1600   # TODO find position after which it does not pick up - depends on enc_vel and robot speed
            pick_list = np.ndarray.tolist(np.asanyarray(pick_list)[is_valid_position])     
            pick_list_positions = pick_list_positions[is_valid_position]

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
            packet_type = packet.item_type

            # Set packet depth to fixed value bz type
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
            if robot_dist > 15: # TODO check value
                state = "WAIT_FOR_PACKET"
                print("state: WAIT_FOR_PACKET")

        # WAIT FOR PACKET
        if state == "WAIT_FOR_PACKET":
            # check encoder and activate robot      # TODO add encoder
            pass # skip for testing with laser start
            state = "PICKING"
            print("state: PICKING")

        if state == "PICKING":
            if is_rob_ready:
                state = "READY"
                print("state: READY")

            

            """                         
            #Trigger start of the pick and place program.
            rc.single_pack_tracking_program_start(
                                        track_result, 
                                        packet, 
                                        encoder_pos, 
                                        encoder_vel, 
                                        is_rob_ready, 
                                        pack_x_offsets, 
                                        pack_depths)
            """
        # Show point cloud visualization when packets are deregistered.
        # if len(deregistered_packets) > 0:
        #     pclv = PointCloudViz(".", deregistered_packets[-1])
        #     pclv.show_point_cloud()
        #     del pclv

        # Show depth frame overlay.
        if depth_map:
            img_detect = cv2.addWeighted(img_detect, 0.8, colorized_depth, 0.3, 0)

        # Show robot position data and FPS.
        if f_data:
            cv2.putText(img_detect,str(robot_server_dict),(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 0), 2)
            cv2.putText(img_detect,
                        "FPS:"+str(1.0/(time.time() - start_time)),
                        (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.57, 
                        (255, 255, 0), 2)

        # Show frames on cv2 window.
        img_detect = cv2.resize(img_detect, (width//2,height//2))
        cv2.imshow("Frame", img_detect)

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

        if key == ord('m') :
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
        
        if key == 27:
            rc.close_program()
            break

def program_mode(rc, rd):
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
            'func':rd.main_robot_control_demo},
        '2':{'dict':Pick_place_dict, 
            'func':rd.main_pick_place},
        '3':{'dict':Pick_place_dict_conv_mov, 
            'func':rd.main_pick_place_conveyor_w_point_cloud},
        '4':{'dict':Pick_place_dict_conv_mov, 
            'func':main}
                }

    # If mode is a program key.
    if mode in modes_dict:
        # Set robot positions and robot program.
        rc.rob_dict = modes_dict[mode]['dict']
        robot_prog = modes_dict[mode]['func']

        # If first mode (not threaded) start program.
        if mode == '1':
            robot_prog(rc)

        # Otherwise start selected threaded program.
        else:
            q = Queue(maxsize = 1)
            t1 = Thread(target = robot_prog, args =(rc, q))
            t2 = Thread(target = rc.robot_server, args =(q, ))
            t1.start()
            t2.start()

    # If input is exit, exit python.
    if mode == 'exit':
        exit()

    # Return function recursively.
    return program_mode(rc, rd)

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
    rc = RobotControl(None)
    rd = RobotDemos(paths, files, check_point)

    # Show message about robot programs.
    print('Select pick and place mode: \n'+
    '1 : Pick and place with static conveyor and hand gestures\n'+
    '2 : Pick and place with static conveyor and multithreading\n'+
    '3 : Pick and place with moving conveyor, point cloud and multithreading\n'+
    '4 : Main Pick and place\n')

    # Start program mode selection.
    program_mode(rc, rd)