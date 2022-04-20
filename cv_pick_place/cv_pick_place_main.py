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
from packet_detection.packet_detector import PacketDetector
from cv2_apriltag.apriltag_detection import ProcessingApriltag
from realsense_config.realsense_depth import DepthCamera
from centroid_tracker.centroidtracker import CentroidTracker
from robot_communication.robot_control import RobotControl

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
check_point ='ckpt-3'
# CUSTOM_MODEL_NAME = 'my_ssd_mobnet_improved_1' 
# check_point ='ckpt-6'

LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {'ANNOTATION_PATH':os.path.join('cv_pick_place',
                                        'Tensorflow',
                                        'workspace',
                                        'annotations'),
        'CHECKPOINT_PATH': os.path.join('cv_pick_place',
                                        'Tensorflow', 
                                        'workspace',
                                        'models',
                                        CUSTOM_MODEL_NAME) 
        }
files = {'PIPELINE_CONFIG':os.path.join('cv_pick_place',
                                        'Tensorflow', 
                                        'workspace',
                                        'models',
                                        CUSTOM_MODEL_NAME,
                                        'pipeline.config'),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], 
                                    LABEL_MAP_NAME)
        }

Pick_place_dict_conv_mov_slow = {
"home_pos" : [{'x':1175.0,'y':267.5,'z':25.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42}],

"pick_pos_base" : [{'x':1175.0,'y':267.5,'z':25.0,
                    'a':90.0,'b':0.0,'c':-180.0,
                    'status':2,'turn':42}],
# place on conveyor points
"place_pos" : [{'x':1420.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1420.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1420.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1420.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42}]
          }

Pick_place_dict_conv_mov = {
"home_pos" : [{'x':1235.0,'y':267.5,'z':165.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42}],

"pick_pos_base" : [{'x':1235.0,'y':267.5,'z':165.0,
                    'a':90.0,'b':0.0,'c':-180.0,
                    'status':2,'turn':42}],
# place on conveyor points
"place_pos" : [{'x':1620.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1620.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1620.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1620.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42}]
          }

Pick_place_dict = {
"home_pos" : [{'x':697.1,'y':0.0,'z':260.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':43}],

"pick_pos_base" : [{'x':368.31,'y':226.34,'z':34.0,
                    'a':90.0,'b':0.0,'c':-180.0,
                    'status':2,'turn':43}],
# place on conveyor points
"place_pos" : [{'x':1079.44,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1250,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1420.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42},

                {'x':1420.73,'y':276.21,'z':45.0,
                'a':90.0,'b':0.0,'c':-180.0,
                'status':2,'turn':42}]
                }
#1 full conveyor rotation = 4527.164 mm in encoder
#encoder circumference = 188.5 mm
# 4527.164/188.5 =~ 24
#gear ratio = 24 ?

def pick():
    """
    Child thread to execute robot pick action.

    """
    rc.Conti_Prog.set_value(ua.DataValue(True))
    time.sleep(0.5)
    rc.Conti_Prog.set_value(ua.DataValue(False))
    print('continue pick')

def robot_server(server_out):
    """
    Thread to get values from PLC server.

    Parameters:
    server_out (object): Queue object where data from PLC server is placed.

    """
    rc.connect_OPCUA_server()
    rc.get_nodes()

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

def main_pick_place_conveyor(server_in):
    """
    Thread for pick and place with moving conveyor.
    
    Parameters:
    server_in (object): Queue object containing data from the PLC server.
    
    """
    apriltag = ProcessingApriltag()
    ct = CentroidTracker(maxDisappeared=10)    
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
    pack_depths = [10.0, 3.0, 5.0, 5.0]
    pack_x_offsets = [100.0,180.0,170.0,170.0]
    while True:
        # print('in size:',server_in.qsize())
        robot_server_dict = server_in.get()
        start_time = time.time()
        rob_stopped = robot_server_dict['rob_stopped']
        stop_active = robot_server_dict['stop_active']
        prog_done = robot_server_dict['prog_done']

        ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
        
        rgb_frame = rgb_frame[:,240:1680]
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
                
        except:
        #Triggered when no markers are in the frame:
            warn_count += 1
            if warn_count == 1:
                print("[INFO]: Markers out of frame or moving.")
            pass
        
        depth_frame = depth_frame[90:400,97:507]
        depth_frame = cv2.resize(depth_frame, (width,height))

        heatmap = colorized_depth
        heatmap = heatmap[90:400,97:507,:]
        heatmap = cv2.resize(heatmap, (width,height))
        
        img_detect, detected = pack_detect.deep_detector_v2(
                                                            rgb_frame, 
                                                            depth_frame, 
                                                            bnd_box = bbox)
        objects = ct.update_detected(detected)
        print(objects, rob_stopped, stop_active, prog_done)
        is_detect = len(detected) is not 0
        encoder_vel = robot_server_dict['encoder_vel']
        is_conv_mov = encoder_vel < - 100.0

        if is_detect:
            if is_conv_mov:
                track_frame += 1
                if track_frame > frames_lim:
                    track_frame = 0
            else:
                track_frame = 0
            track_result = rc.packet_tracking_update(objects, 
                                                    img_detect, 
                                                    homography, 
                                                    is_detect, 
                                                    x_fixed = x_fixed, 
                                                    track_frame = track_frame,
                                                    frames_lim = frames_lim)
            if track_result is not None:
                dist_to_pack = track_result[2]
                delay = dist_to_pack/(abs(encoder_vel)/10)
                delay = round(delay,2)
                # print('delay, distance',delay,dist_to_pack)
                # start_pick = Timer(delay, pick)
                # start_pick.start()
                if  prog_done and (rob_stopped or not stop_active):
                    packet_x = track_result[0]
                    packet_y = track_result[1]
                    angle = detected[0][2]
                    gripper_rot = rc.compute_gripper_rot(angle)
                    packet_type = detected[0][3]
                    print(packet_x,packet_y)
                    rc.change_trajectory(packet_x,
                                        packet_y, 
                                        gripper_rot, 
                                        packet_type,
                                        x_offset = pack_x_offsets[packet_type],
                                        pack_z = pack_depths[packet_type])
                    rc.Start_Prog.set_value(ua.DataValue(True))
                    print('Program Started: ',robot_server_dict['start'])
                    time.sleep(0.5)
                    rc.Start_Prog.set_value(ua.DataValue(False))
                    time.sleep(0.5)

        if depth_map:
            img_detect = cv2.addWeighted(img_detect, 0.8, heatmap, 0.3, 0)

        if f_data:
            cv2.putText(img_detect,str(robot_server_dict),(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 0), 2)
            cv2.putText(img_detect,
                        "FPS:"+str(1.0/(time.time() - start_time)),
                        (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.57, 
                        (255, 255, 0), 2)

        cv2.imshow("Frame", cv2.resize(img_detect, (1280,960)))
        frame_count += 1
        if frame_count == 500:
            frame_count = 1

        key = cv2.waitKey(1)

        if key == ord('o'):
            rc.Gripper_State.set_value(ua.DataValue(False))
            time.sleep(0.1)

        if key == ord('i'):
            rc.Gripper_State.set_value(ua.DataValue(True))
            time.sleep(0.1)

        if key == ord('m') :
            conv_right = not conv_right
            rc.Conveyor_Left.set_value(ua.DataValue(False))
            rc.Conveyor_Right.set_value(ua.DataValue(conv_right))
            time.sleep(0.4)
        
        if key == ord('n'):
            conv_left = not conv_left
            rc.Conveyor_Right.set_value(ua.DataValue(False))
            rc.Conveyor_Left.set_value(ua.DataValue(conv_left))
            time.sleep(0.4)

        if key == ord('l'):
            bbox = not bbox
        
        if key == ord('h'):
            depth_map = not depth_map
                
        if key == ord('f'):
            f_data = not f_data
        
        if key == ord('e'):
            is_detect = not is_detect

        if key == ord('a'):
            rc.Abort_Prog.set_value(ua.DataValue(True))
            print('Program Aborted: ',robot_server_dict['abort'])
            time.sleep(0.5)
        
        if key == ord('c'):
            rc.Conti_Prog.set_value(ua.DataValue(True))
            print('Continue Program')
            time.sleep(0.5)
            rc.Conti_Prog.set_value(ua.DataValue(False))
        
        if key == ord('s'):
            rc.Stop_Prog.set_value(ua.DataValue(True))
            print('Program Interrupted')
            time.sleep(0.5)
            rc.Stop_Prog.set_value(ua.DataValue(False))
        
        if key == 27:
            rc.Abort_Prog.set_value(ua.DataValue(True))
            print('Program Aborted: ',robot_server_dict['abort'])
            rc.Abort_Prog.set_value(ua.DataValue(False))
            rc.Conti_Prog.set_value(ua.DataValue(False))
            rc.client.disconnect()
            cv2.destroyAllWindows()
            print('[INFO]: Client disconnected.')
            time.sleep(0.5)
            break

def program_mode(rc):
    """
    Program selection function.

    Parameters:
    rc (object): RobotControl object for program execution.

    """
    mode = input('Select mode \n'+
    '1 : Pick and place with static conveyor and hand gestures\n'+
    '2 : Pick and place with static conveyor and multithreading\n'+
    '3 : Pick and place with moving conveyor and multithreading\n')
    
    if mode == '1':
        while True:
            rc.rob_dict = Pick_place_dict
            rc.main_robot_control_demo()

    if mode == '2':
        rc.rob_dict = Pick_place_dict
        q = Queue(maxsize = 1)
        t1 = Thread(target = rc.main_pick_place, args =(q, ))
        t2 = Thread(target = robot_server, args =(q, ))
        t1.start()
        t2.start()

    if mode == '3':
        rc.rob_dict = Pick_place_dict_conv_mov
        q = Queue(maxsize = 1)
        t1 = Thread(target = main_pick_place_conveyor, args =(q, ))
        t2 = Thread(target = robot_server, args =(q, ))
        t1.start()
        t2.start()
    else:
        print('Enter valid number')
        program_mode(rc)

if __name__ == '__main__':
    rc = RobotControl(None, paths, files, check_point)
    program_mode(rc)