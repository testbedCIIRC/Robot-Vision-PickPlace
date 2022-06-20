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
import tensorflow as tf
from opcua import Client
import matplotlib as mpl
from scipy import ndimage
from queue import Queue
from threading import Thread
from threading import Timer
from collections import OrderedDict
from scipy.spatial import distance as dist
from cvzone.HandTrackingModule import HandDetector
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.control.robot_control import RobotControl
from robot_cell.control.pick_place_demos import RobotDemos
from robot_cell.packet.centroidtracker import CentroidTracker
from robot_cell.packet.packettracker import PacketTracker
from robot_cell.packet.point_cloud_viz import PointCloudViz

def robot_server(server_out):
    """
    Thread to get values from PLC server.

    Parameters:
    server_out (object): Queue object where data from PLC server is placed.

    """
    # Connect server and get nodes.
    rc.connect_OPCUA_server()
    rc.get_nodes()
    #Enabling laser sensor to synchornize robot with moving packets.
    rc.Laser_Enable.set_value(ua.DataValue(True))
    time.sleep(0.5)
    while True:
        try:
            robot_server_dict = {
            'pos':rc.get_actual_pos(),
            'encoder_vel':round(rc.Encoder_Vel.get_value(),2),
            'encoder_pos':round(rc.Encoder_Pos.get_value(),2),
            'start':rc.Start_Prog.get_value(),
            'abort':rc.Abort_Prog.get_value(),
            'rob_stopped':rc.Rob_Stopped.get_value(),
            'stop_active':rc.Stop_Active.get_value(),
            'prog_done':rc.Prog_Done.get_value()
            }
            server_out.put(robot_server_dict)
            # print('out size:',server_out.qsize())
        except:
            # Triggered when OPCUA server was disconnected.
            print('[INFO]: Queue empty.')
            break

def main(server_in):
    """
    Thread for pick and place with moving conveyor and point cloud operations.
    
    Parameters:
    server_in (object): Queue object containing data from the PLC server.
    
    """
    # Inititalize objects.
    apriltag = ProcessingApriltag()
    pt = PacketTracker(maxDisappeared=10)    
    dc = DepthCamera()
    rc.show_boot_screen('STARTING NEURAL NET...')
    pack_detect = PacketDetector(rc.paths, rc.files, rc.checkpt)

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

        # Get frames from realsense.
        ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
        
        # Crop frame to rgb frame to 1080x1440x3.
        rgb_frame = rgb_frame[:,240:1680]

        # Crop and resize depth frame to match rgb frame.
        height, width, depth = rgb_frame.shape
        depth_frame = depth_frame[90:400,97:507]
        depth_frame = cv2.resize(depth_frame, (width,height))

        # Crop and resize colorized depth frame to match rgb frame.
        heatmap = colorized_depth
        heatmap = heatmap[90:400,97:507,:]
        heatmap = cv2.resize(heatmap, (width,height))

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

            # Reset not detected tags warning.
            if is_marker_detect:
                warn_count = 0
                
        #Triggered when no markers are in the frame.
        except Exception as e:
            warn_count += 1
            # Print warning only once.
            if warn_count == 1:
                print(e)
                print("[INFO]: Markers out of frame or moving.")
            pass
        
        # Detect packets using neural network.
        img_detect, detected = pack_detect.deep_pack_obj_detector(
                                                            rgb_frame, 
                                                            depth_frame,
                                                            encoder_pos, 
                                                            bnd_box = bbox)
        # Update tracked packets for current frame.
        registered_packets, deregistered_packets = pt.update(detected, depth_frame)
        print({
            'regis_packs': registered_packets,
            'rob_stopped': rob_stopped, 
            'stop_active': stop_active, 
            'prog_done': prog_done})

        # When detected not empty, objects are being detected.
        is_detect = len(detected) != 0 
        # When speed of conveyor more than -100 it is moving to the left.
        is_conv_mov = encoder_vel < - 100.0
        #Robot ready when programs are fully finished and it isn't moving.
        is_rob_ready = prog_done and (rob_stopped or not stop_active)

        # If packets are being tracked.
        if is_detect:
            # If the conveyor is moving to the left direction.
            if is_conv_mov:
                # Increase counter of frames with detections. 
                track_frame += 1
                # If counter larger than limit.
                if track_frame > frames_lim:
                    # Reset tracked frames count.
                    track_frame = 0
            # If conveyor stops moving to the left direction.
            else:
                # Set tracked frames count to 0.
                track_frame = 0
            
            # Compute updated (x,y) pick positions of tracked moving packets and distance to packet.
            track_result, packet = rc.pack_obj_tracking_update(
                                                        registered_packets, 
                                                        img_detect, 
                                                        homography, 
                                                        is_detect,
                                                        x_fixed, 
                                                        track_frame,
                                                        frames_lim,
                                                        encoder_pos)
            #Trigger start of the pick and place program.
            rc.pack_obj_tracking_program_start(
                                        track_result, 
                                        packet, 
                                        encoder_pos, 
                                        encoder_vel, 
                                        is_rob_ready, 
                                        pack_x_offsets, 
                                        pack_depths)

        # Show point cloud visualization when packets are deregistered.
        if len(deregistered_packets) > 0:
            pclv = PointCloudViz("temp_rgbd", deregistered_packets[-1])
            pclv.show_point_cloud()
            del pclv

        # Show depth frame overlay.
        if depth_map:
            img_detect = cv2.addWeighted(img_detect, 0.8, heatmap, 0.3, 0)

        # Show robot position data and FPS.
        if f_data:
            cv2.putText(img_detect,str(robot_server_dict),(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 0), 2)
            cv2.putText(img_detect,
                        "FPS:"+str(1.0/(time.time() - start_time)),
                        (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.57, 
                        (255, 255, 0), 2)

        # Show frames on cv2 window.
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Frame", cv2.resize(img_detect, (1280,960)))

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

def program_mode(rc):
    """
    Program selection function.

    Parameters:
    rc (object): RobotControl object for program execution.

    """
    # Read mode input.
    mode = input()
    # Dictionary with robot positions and robot programs.
    modes_dict = {
        '1':{'dict':Pick_place_dict, 'func':rc.main_robot_control_demo},
        '2':{'dict':Pick_place_dict, 'func':rc.main_pick_place},
        '3':{'dict':Pick_place_dict_conv_mov, 'func':rc.main_pick_place_conveyor},
        '4':{'dict':Pick_place_dict_conv_mov, 'func':rc.main_pick_place_conveyor_w_point_cloud},
        '5':{'dict':Pick_place_dict_conv_mov, 'func':main}}

    # If mode is a program key.
    if mode in modes_dict:
        # Set robot positions and robot program.
        rc.rob_dict = modes_dict[mode]['dict']
        robot_prog = modes_dict[mode]['func']

        # If first mode (not threaded) start program in loop.
        if mode == '1':
            while True:
                robot_prog()

        # Otherwise start selected threaded program.
        else:
            q = Queue(maxsize = 1)
            t1 = Thread(target = robot_prog, args =(q, ))
            t2 = Thread(target = robot_server, args =(q, ))
            t1.start()
            t2.start()

    # If input is exit, exit python.
    if mode == 'exit':
        exit()

    # Return function recursively.
    return program_mode(rc)

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
    
    # Initialize robot demos object.
    rc = RobotDemos(None, paths, files, check_point)

    # Show message about robot programs.
    print('Select pick and place mode: \n'+
    '1 : Pick and place with static conveyor and hand gestures\n'+
    '2 : Pick and place with static conveyor and multithreading\n'+
    '3 : Pick and place with moving conveyor and multithreading\n'+
    '4 : Pick and place with moving conveyor, point cloud and multithreading\n'+
    '5 : Main Pick and place\n')

    # Start program mode selection.
    program_mode(rc)