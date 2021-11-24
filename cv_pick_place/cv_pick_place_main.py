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
paths = {
    'ANNOTATION_PATH': os.path.join('cv_pick_place','Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('cv_pick_place','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME) 
}
files = {
    'PIPELINE_CONFIG':os.path.join('cv_pick_place','Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

Pick_place_dict = {
"home_pos":[{'x':697.1,'y':0.0,'z':260.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':43}],

"pick_pos_base": [{'x':368.31,'y':226.34,'z':34.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':43}],

# place on conveyor points
"place_pos":[{'x':1079.44,'y':276.21,'z':45.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
            {'x':1250,'y':276.21,'z':45.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
            {'x':1420.73,'y':276.21,'z':45.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
            {'x':1420.73,'y':276.21,'z':45.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42}]
          }
#place on boxes points
# "place_pos":[{'x':1704.34,'y':143.92,'z':295.65,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
#             {'x':944.52,'y':124.84,'z':177.56,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
#             {'x':1284.27,'y':145.21,'z':274.95,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
#             {'x':1284.27,'y':145.21,'z':274.95,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42}]
#           }
def robot_server(server_out):
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
            'rob_stopped':rc.Rob_Stopped.get_value()
            }
            server_out.put(robot_server_dict)
            # print('out size:',server_out.qsize())
        except:
            print('[INFO]: Queue empty.')
            break

def main_robot_control(server_in):
    apriltag = ProcessingApriltag(None, None, None)
    ct = CentroidTracker()    
    dc = DepthCamera()    
    rc.show_boot_screen('STARTING NEURAL NET...')
    pack_detect = PacketDetector(rc.paths, rc.files, rc.checkpt)
    warn_count = 0
    frames_lim = 0
    is_detect = False
    conv_left = False
    conv_right = False
    bbox = True
    depth_map = True
    f_data = False
    homography = None
    while True:
        # print('in size:',server_in.qsize())
        robot_server_dict = server_in.get()
        start_time = time.time()
        rob_stopped = robot_server_dict['rob_stopped']

        ret, depth_frame, color_frame, colorized_depth = dc.get_frame()
        
        color_frame = color_frame[:,240:1680]
        height, width, depth = color_frame.shape[0],color_frame.shape[1],color_frame.shape[2]
        
        try:
            color_frame = apriltag.detect_tags(color_frame)
            homography = apriltag.compute_homog()

            is_marker_detect= type(homography).__module__ == np.__name__ or homography == None
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
        
        img_np_detect, result, rects = pack_detect.deep_detector(color_frame, depth_frame, homography, bnd_box = bbox)
        
        objects = ct.update(rects)
        # print(objects)
        # rc.objects_update(objects, img_np_detect)
        if is_detect:
            frames_lim += 1
            if frames_lim == 21:
                frames_lim = 0
        rc.packet_tracking_update(objects, img_np_detect, homography, is_detect, x_fixed = 0, frames_lim = frames_lim)
        
        if depth_map:
            img_np_detect = cv2.addWeighted(img_np_detect, 0.8, heatmap, 0.3, 0)

        if f_data:
            x_pos, y_pos, z_pos, a_pos, b_pos, c_pos, status_pos, turn_pos = robot_server_dict['pos']
            encoder_vel = robot_server_dict['encoder_vel']
            encoder_pos = robot_server_dict['encoder_pos']
            cv2.putText(img_np_detect,str(robot_server_dict),(10,25),cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 0), 2)

            print("FPS: ", 1.0 / (time.time() - start_time))

        cv2.imshow("Frame", cv2.resize(img_np_detect, (1280,960)))

        key = cv2.waitKey(1)

        if rob_stopped:
            if key == ord('b'):
                bpressed += 1
                if bpressed == 5:
                    print(rects)
                    world_centroid = rects[0][2]
                    packet_x = round(world_centroid[0] * 10.0, 2)
                    packet_y = round(world_centroid[1] * 10.0, 2)
                    angle = rects[0][3]
                    gripper_rot = rc.compute_gripper_rot(angle)
                    packet_type = rects[0][4]
                    rc.change_trajectory(packet_x, packet_y, gripper_rot, packet_type)
                    rc.Start_Prog.set_value(ua.DataValue(True))
                    print('Program Started: ',robot_server_dict['start'])
                    rc.Start_Prog.set_value(ua.DataValue(False))
                    time.sleep(0.5)
                    bpressed = 0
            elif key != ord('b'):
                bpressed = 0

        if key == ord('o'):
            rc.Gripper_State.set_value(ua.DataValue(False))
            time.sleep(0.1)

        if key == ord('i'):
            rc.Gripper_State.set_value(ua.DataValue(True))
            time.sleep(0.1)

        if key == ord('n') :
            conv_right = not conv_right
            rc.Conveyor_Right.set_value(ua.DataValue(conv_right))
            time.sleep(0.1)
        
        if key == ord('m'):
            conv_left = not conv_left
            rc.Conveyor_Left.set_value(ua.DataValue(conv_left))
            time.sleep(0.1)

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
            rc.client.disconnect()
            cv2.destroyAllWindows()
            print('[INFO]: Client disconnected.')
            time.sleep(0.5)
            break

if __name__ == '__main__':
    # while True:
        # rc = RobotControl(Pick_place_dict, paths, files, check_point)
        # rc.main_robot_control_demo()
    # rc = RobotControl(Pick_place_dict, paths, files, check_point)
    # rc.main_pick_place()

    rc = RobotControl(Pick_place_dict, paths, files, check_point)
    q = Queue(maxsize = 1)
    t1 = Thread(target = main_robot_control, args =(q, ))
    t2 = Thread(target = robot_server, args =(q, ))
    t1.start()
    t2.start()
        
