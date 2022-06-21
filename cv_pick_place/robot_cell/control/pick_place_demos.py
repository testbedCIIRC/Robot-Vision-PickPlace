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
from robot_cell.packet.centroidtracker import CentroidTracker
from robot_cell.packet.packettracker import PacketTracker
from robot_cell.packet.point_cloud_viz import PointCloudViz

class RobotDemos(RobotControl):
    def __init__(self, rob_dict, paths, files, checkpt):
        """
        RobotDemos object constructor.
    
        Parameters:
        rob_dict (dict): Dictionary with robot points for program.
        paths (dict): Dictionary with annotation and checkpoint paths.
        files (dict): Dictionary with pipeline and config paths. 

        """
        self.paths = paths
        self.files = files
        self.checkpt = checkpt
        super().__init__(rob_dict)

    def gripper_gesture_control(self, detector, cap, show = False):
        """
        Function used to control the gripper with hand gestures.

        Parameters:
        detector (object): Detector object from cvzone library.
        cap (object): A cv2.VideoCapture object to access webcam.
        show (bool): Boolean to enable or disable the function.

        """
        success, img = cap.read()
        if show:
            hands, img = detector.findHands(img)
                # # without draw
                # hands = detector.findHands(img, draw=False)
            if hands:
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"] 
                bbox1 = hand1["bbox"]  
                centerPoint1 = hand1['center']  
                handType1 = hand1["type"]  

                fingers1 = detector.fingersUp(hand1)

                if len(hands) == 2:
                    # Hand 2
                    hand2 = hands[1]
                    # List of 21 Landmark points
                    lmList2 = hand2["lmList"]  
                    # Bounding box info x,y,w,h
                    bbox2 = hand2["bbox"]  
                    # center of the hand cx,cy
                    centerPoint2 = hand2['center']  
                    # Hand Type "Left" or "Right"
                    handType2 = hand2["type"]  

                    fingers2 = detector.fingersUp(hand2)
                    # Find Distance between two Landmarks. 
                    # Could be same hand or different hands
                    length, info, img = detector.findDistance(lmList1[8], 
                                                                lmList2[8], 
                                                                img)
                    # with draw
                    length, info = detector.findDistance(lmList1[8], lmList2[8])
                    if length <=30.0:
                        self.Gripper_State.set_value(ua.DataValue(False))
                        time.sleep(0.1)
                    if length >=100.0:
                        self.Gripper_State.set_value(ua.DataValue(True))
                        time.sleep(0.1)
        else:
            img = np.zeros_like(img)

        cv2.imshow("Gestures", img)

    def main_packet_detect(self):
            """
            Basic main packet detection.
        
            Returns:
            tuple(np.ndarray, list): Image  with detections and detections.
        
            """
            self.show_boot_screen('STARTING NEURAL NET...')
            warn_count = 0
            a = 0
            b = 0
            d = 2.61
            # d = 3
            bbox = True
            f_rate = False
            ct = CentroidTracker()    
            dc = DepthCamera()    
            apriltag = ProcessingApriltag()
            pack_detect = PacketDetector(self.paths, self.files, self.checkpt)
            homography = None
            while True:
                start_time = time.time()
                ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
                
                rgb_frame = rgb_frame[:,240:1680]
                # rgb_frame = cv2.resize(rgb_frame, (640,480))
                height, width, depth = rgb_frame.shape
                
                try:
                    rgb_frame = apriltag.detect_tags(rgb_frame)
                    homography = apriltag.compute_homog()
                    is_type_np = type(homography).__module__ == np.__name__
                    is_marker_detect = is_type_np or homography == None
                    if is_marker_detect:
                        warn_count = 0
                        # print(homography)         
                except:
                #Triggered when no markers are in the frame:
                    warn_count += 1
                    if warn_count == 1:
                        print("[INFO]: Markers out of frame or moving.")
                    pass

                # rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=a, beta=b)
                # print(a,b,d)
                
                depth_frame = depth_frame[90:400,97:507]
                depth_frame = cv2.resize(depth_frame, (width,height))

                # heatmap = cv2.applyColorMap(np.uint8(depth_frame*d), 
                                            # cv2.COLORMAP_TURBO)
                heatmap = colorized_depth
                heatmap = heatmap[90:400,97:507,:]
                heatmap = cv2.resize(heatmap, (width,height))
                
                img_detect, detected = pack_detect.deep_detector(rgb_frame, 
                                                                        depth_frame, 
                                                                        homography, 
                                                                        bnd_box = bbox)
                
                objects = ct.update(detected)
                self.objects_update(objects, img_detect)

                cv2.circle(img_detect, (int(width/2),int(height/2) ), 
                            4, (0, 0, 255), -1)
                added_image = cv2.addWeighted(img_detect, 0.8, heatmap, 0.3, 0)
                if f_rate:
                    cv2.putText(added_image,
                                'FPS:'+ str( 1.0 / (time.time() - start_time)),
                                (60,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                (255, 255, 0), 2)
                    print("FPS: ", 1.0 / (time.time() - start_time))
                
                cv2.imshow("Frame", cv2.resize(added_image, (1280,960)))

                key = cv2.waitKey(1)
                if key== ord('w'):
                    a+=0.1
                if key== ord('s'):
                    a-=0.1
                if key== ord('a'):
                    b+=1
                if key== ord('d'):
                    b-=1
                if key== ord('z'):
                    d+=2
                if key== ord('x'):
                    d-=2
                if key == ord('l'):
                    bbox = not bbox
                if key == ord('f'):
                    f_rate = not f_rate
                if key== 27:
                    # cv2.destroyAllWindows()
                    break
            print(detected)
            return added_image , detected

    def main_robot_control_demo(self):
        """
        Pick and place with static conveyor and hand gestures.

        """
        detected_img, detected = self.main_packet_detect()

        self.connect_OPCUA_server()

        world_centroid = detected[0][2]
        packet_x = round(world_centroid[0] * 10.0, 2)
        packet_y = round(world_centroid[1] * 10.0, 2)
        angle = detected[0][3]
        gripper_rot = self.compute_gripper_rot(angle)
        packet_type = detected[0][4]

        self.get_nodes()

        frame_num = -1
        bpressed = 0
        dc = DepthCamera()
        gripper_ON = self.Gripper_State.get_value()
        cap = cv2.VideoCapture(2)
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        show_gestures = True
        while True:
            start = self.Start_Prog.get_value()
            rob_stopped = self.Rob_Stopped.get_value()
            abort = self.Abort_Prog.get_value()
            encoder_vel = self.Encoder_Vel.get_value()
            encoder_pos = self.Encoder_Pos.get_value()

            x_pos = self.Act_Pos_X.get_value()
            y_pos = self.Act_Pos_Y.get_value()
            z_pos = self.Act_Pos_Z.get_value()
            a_pos = self.Act_Pos_A.get_value()
            b_pos = self.Act_Pos_B.get_value()
            c_pos =self.Act_Pos_C.get_value()
            status_pos = self.Act_Pos_Status.get_value()
            turn_pos = self.Act_Pos_Turn.get_value()

            prePick_done = self.PrePick_Done.get_value()
            place_done = self.Place_Done.get_value()

            ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
            rgb_frame = rgb_frame[:,240:1680]
            frame_num += 1
            height, width, depth = rgb_frame.shape
            # rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.2, beta=10)
            
            self.gripper_gesture_control(detector, cap, show = show_gestures)
            
            x_pos = round(x_pos,2)
            y_pos = round(y_pos,2)
            z_pos = round(z_pos,2)
            a_pos = round(a_pos,2)
            b_pos = round(b_pos,2)
            c_pos = round(c_pos,2)
            encoder_vel = round(encoder_vel,2)
            encoder_pos = round(encoder_pos,2)

            cv2.circle(rgb_frame, (int(width/2),int(height/2) ), 
                        4, (0, 0, 255), -1)
            cv2.putText(rgb_frame,'x:'+ str(x_pos),(60,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'y:'+ str(y_pos),(60,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'z:'+ str(z_pos),(60,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'a:'+ str(a_pos),(60,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'b:'+ str(b_pos),(60,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'c:'+ str(c_pos),(60,130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'Status:'+ str(status_pos),(60,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'Turn:'+ str(turn_pos),(60,170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'Enc. Speed:'+ str(encoder_vel),(60,190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(rgb_frame,'Enc. Position:'+ str(encoder_pos),(60,210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            rgb_frame = cv2.addWeighted(rgb_frame, 0.8, detected_img, 0.4, 0)

            cv2.imshow("Frame", cv2.resize(rgb_frame, (1280,960)))
            cv2.imshow('Object detected', cv2.resize(detected_img, (1280,960)))

            key = cv2.waitKey(1)
            if key == 27:
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',abort)
                self.Abort_Prog.set_value(ua.DataValue(False))
                cap.release()
                self.client.disconnect()
                # self.gripper_gesture_control(detector, cap, show = False)
                cv2.destroyWindow("Gestures")
                cv2.destroyWindow("Object detected")
                time.sleep(0.5)
                break

            if rob_stopped:
                if key == ord('b'):
                    bpressed += 1
                    if bpressed == 5:
                        self.change_trajectory(packet_x, 
                                                packet_y, 
                                                gripper_rot, 
                                                packet_type)
                        self.Start_Prog.set_value(ua.DataValue(True))
                        print('Program Started: ',start)
                        self.Start_Prog.set_value(ua.DataValue(False))
                        time.sleep(0.5)
                        bpressed = 0
                elif key != ord('b'):
                    bpressed = 0

            if key == ord('o'):
                self.Gripper_State.set_value(ua.DataValue(False))
                time.sleep(0.1)

            if key == ord('i'):
                self.Gripper_State.set_value(ua.DataValue(True))
                time.sleep(0.1)
            
            if key == ord('g'):
                show_gestures = not show_gestures
                
            if key == ord('a'):
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',abort)
                time.sleep(0.5)
                
    def main_pick_place(self, server_in):
        """
        Pick and place with static conveyor and multithreading.

        Parameters:
        server_in (object): Queue object containing data from the PLC server.

        """
        apriltag = ProcessingApriltag()
        ct = CentroidTracker()    
        dc = DepthCamera()    
        self.show_boot_screen('STARTING NEURAL NET...')
        pack_detect = PacketDetector(self.paths, self.files, self.checkpt)
        warn_count = 0
        track_frame = 0
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
            stop_active = robot_server_dict['stop_active']
            prog_done = robot_server_dict['prog_done']

            ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
            
            rgb_frame = rgb_frame[:,240:1680]
            height, width, depth = rgb_frame.shape
            
            try:
                rgb_frame = apriltag.detect_tags(rgb_frame)
                homography = apriltag.compute_homog()
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
            
            img_detect, detected = pack_detect.deep_detector(rgb_frame, 
                                                                    depth_frame, 
                                                                    homography, 
                                                                    bnd_box = bbox)
            
            objects = ct.update(detected)
            # print(objects)
            self.objects_update(objects, img_detect)
            
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

            key = cv2.waitKey(1)

            if prog_done and (rob_stopped or not stop_active):
                if key == ord('b'):
                    bpressed += 1
                    if bpressed == 5:
                        print(detected)
                        world_centroid = detected[0][2]
                        packet_x = round(world_centroid[0] * 10.0, 2)
                        packet_y = round(world_centroid[1] * 10.0, 2)
                        angle = detected[0][3]
                        gripper_rot = self.compute_gripper_rot(angle)
                        packet_type = detected[0][4]
                        self.change_trajectory(packet_x, 
                                                packet_y, 
                                                gripper_rot, 
                                                packet_type)
                        self.Start_Prog.set_value(ua.DataValue(True))
                        print('Program Started: ',robot_server_dict['start'])
                        self.Start_Prog.set_value(ua.DataValue(False))
                        time.sleep(0.5)
                        bpressed = 0
                elif key != ord('b'):
                    bpressed = 0

            if key == ord('o'):
                self.Gripper_State.set_value(ua.DataValue(False))
                time.sleep(0.1)

            if key == ord('i'):
                self.Gripper_State.set_value(ua.DataValue(True))
                time.sleep(0.1)

            if key == ord('m') :
                conv_right = not conv_right
                self.Conveyor_Right.set_value(ua.DataValue(conv_right))
                time.sleep(0.1)
            
            if key == ord('n'):
                conv_left = not conv_left
                self.Conveyor_Left.set_value(ua.DataValue(conv_left))
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
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',robot_server_dict['abort'])
                time.sleep(0.5)
            
            if key == ord('c'):
                self.Conti_Prog.set_value(ua.DataValue(True))
                print('Continue Program')
                time.sleep(0.5)
                self.Conti_Prog.set_value(ua.DataValue(False))
            
            if key == ord('s'):
                self.Stop_Prog.set_value(ua.DataValue(True))
                print('Program Interrupted')
                time.sleep(0.5)
                self.Stop_Prog.set_value(ua.DataValue(False))
            
            if key == 27:
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',robot_server_dict['abort'])
                self.Abort_Prog.set_value(ua.DataValue(False))
                self.client.disconnect()
                cv2.destroyAllWindows()
                print('[INFO]: Client disconnected.')
                time.sleep(0.5)
                break
    def main_pick_place_conveyor(self, server_in):
        """
        Thread for pick and place with moving conveyor.
        
        Parameters:
        server_in (object): Queue object containing data from the PLC server.
        
        """
        apriltag = ProcessingApriltag()
        ct = CentroidTracker(maxDisappeared=10)    
        dc = DepthCamera()    
        self.show_boot_screen('STARTING NEURAL NET...')
        pack_detect = PacketDetector(self.paths, self.files, self.checkpt)
        x_fixed = self.rob_dict['pick_pos_base'][0]['x']
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
            is_detect = len(detected) != 0
            encoder_vel = robot_server_dict['encoder_vel']
            is_conv_mov = encoder_vel < - 100.0

            if is_detect:
                if is_conv_mov:
                    track_frame += 1
                    if track_frame > frames_lim:
                        track_frame = 0
                else:
                    track_frame = 0
                track_result = self.packet_tracking_update(objects, 
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
                        gripper_rot = self.compute_gripper_rot(angle)
                        packet_type = detected[0][3]
                        print(packet_x,packet_y)
                        self.change_trajectory(packet_x,
                                            packet_y, 
                                            gripper_rot, 
                                            packet_type,
                                            x_offset = pack_x_offsets[packet_type],
                                            pack_z = pack_depths[packet_type])
                        self.Start_Prog.set_value(ua.DataValue(True))
                        print('Program Started: ',robot_server_dict['start'])
                        time.sleep(0.5)
                        self.Start_Prog.set_value(ua.DataValue(False))
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
                self.Gripper_State.set_value(ua.DataValue(False))
                time.sleep(0.1)

            if key == ord('i'):
                self.Gripper_State.set_value(ua.DataValue(True))
                time.sleep(0.1)

            if key == ord('m') :
                conv_right = not conv_right
                self.Conveyor_Left.set_value(ua.DataValue(False))
                self.Conveyor_Right.set_value(ua.DataValue(conv_right))
                time.sleep(0.4)
            
            if key == ord('n'):
                conv_left = not conv_left
                self.Conveyor_Right.set_value(ua.DataValue(False))
                self.Conveyor_Left.set_value(ua.DataValue(conv_left))
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
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',robot_server_dict['abort'])
                time.sleep(0.5)
            
            if key == ord('c'):
                self.Conti_Prog.set_value(ua.DataValue(True))
                print('Continue Program')
                time.sleep(0.5)
                self.Conti_Prog.set_value(ua.DataValue(False))
            
            if key == ord('s'):
                self.Stop_Prog.set_value(ua.DataValue(True))
                print('Program Interrupted')
                time.sleep(0.5)
                self.Stop_Prog.set_value(ua.DataValue(False))
            
            if key == 27:
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',robot_server_dict['abort'])
                self.Abort_Prog.set_value(ua.DataValue(False))
                self.Conti_Prog.set_value(ua.DataValue(False))
                self.client.disconnect()
                cv2.destroyAllWindows()
                print('[INFO]: Client disconnected.')
                time.sleep(0.5)
                break
    def main_pick_place_conveyor_w_point_cloud(self, server_in):
        """
        Thread for pick and place with moving conveyor and point cloud operations.
        
        Parameters:
        server_in (object): Queue object containing data from the PLC server.
        
        """
        apriltag = ProcessingApriltag()
        pt = PacketTracker(maxDisappeared=10)    
        dc = DepthCamera()    
        self.show_boot_screen('STARTING NEURAL NET...')
        pack_detect = PacketDetector(self.paths, self.files, self.checkpt)
        x_fixed = self.rob_dict['pick_pos_base'][0]['x']
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

            if is_detect:
                if is_conv_mov:
                    track_frame += 1
                    if track_frame > frames_lim:
                        track_frame = 0
                else:
                    track_frame = 0
                track_result, packet = self.pack_obj_tracking_update(objects, 
                                                        img_detect, 
                                                        homography, 
                                                        is_detect,
                                                        x_fixed, 
                                                        track_frame,
                                                        frames_lim,
                                                        encoder_pos) 
                if track_result is not None:
                    dist_to_pack = track_result[2]
                    delay = dist_to_pack/(abs(encoder_vel)/10)
                    delay = round(delay,2)
                    if  prog_done and (rob_stopped or not stop_active):
                        packet_x = track_result[0]
                        packet_y = track_result[1]
                        angle = packet.angle
                        gripper_rot = self.compute_gripper_rot(angle)
                        packet_type = packet.pack_type
                        print(packet_x,packet_y)
                        self.change_trajectory(packet_x,
                                            packet_y, 
                                            gripper_rot, 
                                            packet_type,
                                            x_offset = pack_x_offsets[packet_type],
                                            pack_z = pack_depths[packet_type])
                        self.Start_Prog.set_value(ua.DataValue(True))
                        print('Program Started: ',robot_server_dict['start'])
                        time.sleep(0.5)
                        self.Start_Prog.set_value(ua.DataValue(False))
                        time.sleep(0.5)

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
                self.Gripper_State.set_value(ua.DataValue(False))
                time.sleep(0.1)

            if key == ord('i'):
                self.Gripper_State.set_value(ua.DataValue(True))
                time.sleep(0.1)

            if key == ord('m') :
                conv_right = not conv_right
                self.Conveyor_Left.set_value(ua.DataValue(False))
                self.Conveyor_Right.set_value(ua.DataValue(conv_right))
                time.sleep(0.4)
            
            if key == ord('n'):
                conv_left = not conv_left
                self.Conveyor_Right.set_value(ua.DataValue(False))
                self.Conveyor_Left.set_value(ua.DataValue(conv_left))
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
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',robot_server_dict['abort'])
                time.sleep(0.5)
            
            if key == ord('c'):
                self.Conti_Prog.set_value(ua.DataValue(True))
                print('Continue Program')
                time.sleep(0.5)
                self.Conti_Prog.set_value(ua.DataValue(False))
            
            if key == ord('s'):
                self.Stop_Prog.set_value(ua.DataValue(True))
                print('Program Interrupted')
                time.sleep(0.5)
                self.Stop_Prog.set_value(ua.DataValue(False))
            
            if key == 27:
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',robot_server_dict['abort'])
                self.Abort_Prog.set_value(ua.DataValue(False))
                self.Conti_Prog.set_value(ua.DataValue(False))
                self.client.disconnect()
                cv2.destroyAllWindows()
                print('[INFO]: Client disconnected.')
                time.sleep(0.5)
                break