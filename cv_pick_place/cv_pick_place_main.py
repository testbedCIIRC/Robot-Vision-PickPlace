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
    rc.connect_OPCUA_server()
    rc.get_nodes()
    
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
            print('[INFO]: Queue empty.')
            break

def main(server_in):
    """
    Thread for pick and place with moving conveyor and point cloud operations.
    
    Parameters:
    server_in (object): Queue object containing data from the PLC server.
    
    """
    apriltag = ProcessingApriltag()
    pt = PacketTracker(maxDisappeared=10)    
    dc = DepthCamera()    
    rc.show_boot_screen('STARTING NEURAL NET...')
    pack_detect = PacketDetector(rc.paths, rc.files, rc.checkpt)
    x_fixed = rc.rob_dict['pick_pos_base'][0]['x']
    warn_count = 0
    track_frame = 0
    frames_lim = 10
    bbox = True
    f_data = False
    depth_map = True
    is_detect = False
    conv_left = False
    conv_right = False
    frame_count = 1
    homography = None
    track_result = None
    #with speed 55% :
    pack_depths = [10.0, 3.0, 5.0, 5.0]
    pack_x_offsets = [50.0,180.0,130.0,130.0]
    
    while True:
        # print('in size:',server_in.qsize())
        robot_server_dict = server_in.get()
        start_time = time.time()
        rob_stopped = robot_server_dict['rob_stopped']
        stop_active = robot_server_dict['stop_active']
        prog_done = robot_server_dict['prog_done']

        ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
        
        rgb_frame = rgb_frame[:,240:1680]
        # 1080x1440x3
        height, width, depth = rgb_frame.shape
        try:
            rgb_frame = apriltag.detect_tags(rgb_frame)
            if frame_count == 1:
                homography = apriltag.compute_homog()
                print('[INFO]: Homography matrix updated.')
            is_type_np = type(homography).__module__ == np.__name__
            is_marker_detect = is_type_np or homography == None
            if is_marker_detect:
                warn_count = 0
                
        except Exception as e:
        #Triggered when no markers are in the frame:
            warn_count += 1
            if warn_count == 1:
                print(e)
                print("[INFO]: Markers out of frame or moving.")
            pass
        
        depth_frame = depth_frame[90:400,97:507]
        depth_frame = cv2.resize(depth_frame, (width,height))

        heatmap = colorized_depth
        heatmap = heatmap[90:400,97:507,:]
        heatmap = cv2.resize(heatmap, (width,height))
        encoder_vel = robot_server_dict['encoder_vel']
        encoder_pos = robot_server_dict['encoder_pos']
        
        img_detect, detected = pack_detect.deep_pack_obj_detector(
                                                            rgb_frame, 
                                                            depth_frame,
                                                            encoder_pos, 
                                                            bnd_box = bbox)
        objects, deregistered_packets = pt.update(detected, depth_frame)
        print(objects, rob_stopped, stop_active, prog_done)
        is_detect = len(detected) != 0
        is_conv_mov = encoder_vel < - 100.0
        is_rob_ready = prog_done and (rob_stopped or not stop_active)

        if is_detect:
            if is_conv_mov:
                track_frame += 1
                if track_frame > frames_lim:
                    track_frame = 0
            else:
                track_frame = 0

            track_result, packet = rc.pack_obj_tracking_update(objects, 
                                                    img_detect, 
                                                    homography, 
                                                    is_detect,
                                                    x_fixed, 
                                                    track_frame,
                                                    frames_lim,
                                                    encoder_pos)

            rc.pack_obj_tracking_program_start(track_result, 
                                                packet, 
                                                encoder_pos, 
                                                encoder_vel, 
                                                is_rob_ready, 
                                                pack_x_offsets, 
                                                pack_depths)

        if len(deregistered_packets) > 0:
            pclv = PointCloudViz("temp_rgbd", deregistered_packets[-1])
            pclv.show_point_cloud()
            del pclv

        if depth_map:
            img_detect = cv2.addWeighted(img_detect, 0.8, heatmap, 0.3, 0)

        if f_data:
            cv2.putText(img_detect,str(robot_server_dict),(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 0), 2)
            cv2.putText(img_detect,
                        "FPS:"+str(1.0/(time.time() - start_time)),
                        (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.57, 
                        (255, 255, 0), 2)

        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Frame", cv2.resize(img_detect, (1280,960)))
        
        frame_count += 1
        if frame_count == 500:
            frame_count = 1

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
    mode = input()
    thread_modes = ['2','3','4','5']
    if mode == '1':
        while True:
            rc.rob_dict = Pick_place_dict
            rc.main_robot_control_demo()

    if mode in thread_modes:
        q = Queue(maxsize = 1)

        if mode == '2':
            rc.rob_dict = Pick_place_dict
            t1 = Thread(target = rc.main_pick_place, args = (q, ))

        if mode == '3':
            rc.rob_dict = Pick_place_dict_conv_mov
            t1 = Thread(target = rc.main_pick_place_conveyor, args = (q, ))

        if mode == '4':
            rc.rob_dict = Pick_place_dict_conv_mov
            t1 = Thread(target = rc.main_pick_place_conveyor_w_point_cloud, args = (q, ))
        
        if mode == '5':
            rc.rob_dict = Pick_place_dict_conv_mov
            t1 = Thread(target = main, args =(q, ))
            
        t2 = Thread(target = robot_server, args =(q, ))
        t1.start()
        t2.start()

    if mode == 'exit':
        exit()

    return program_mode(rc)

if __name__ == '__main__':
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    check_point ='ckpt-3'
    # CUSTOM_MODEL_NAME = 'my_ssd_mobnet_improved_1' 
    # check_point ='ckpt-6'

    LABEL_MAP_NAME = 'label_map.pbtxt'
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

    file = open('robot_positions.json')
    robot_poses = json.load(file)

    Pick_place_dict_conv_mov_slow = robot_poses['Pick_place_dict_conv_mov_slow']
    Pick_place_dict_conv_mov = robot_poses['Pick_place_dict_conv_mov']
    Pick_place_dict = robot_poses['Pick_place_dict']
    
    rc = RobotDemos(None, paths, files, check_point)
    print('Select mode \n'+
    '1 : Pick and place with static conveyor and hand gestures\n'+
    '2 : Pick and place with static conveyor and multithreading\n'+
    '3 : Pick and place with moving conveyor and multithreading\n'+
    '4 : Pick and place with moving conveyor, point cloud and multithreading\n'+
    '5 : Main Pick and place\n')
    program_mode(rc)