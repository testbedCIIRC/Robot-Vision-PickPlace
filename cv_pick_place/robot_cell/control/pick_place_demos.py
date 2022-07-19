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

from robot_cell.control.robot_control import RobotControl
from robot_cell.packet.packettracker import PacketTracker
from robot_cell.packet.point_cloud_viz import PointCloudViz
from robot_cell.packet.centroidtracker import CentroidTracker
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag

class RobotDemos:
    def __init__(self, paths, files, checkpt):
        """
        RobotDemos object constructor.
    
        Parameters:
        paths (dict): Dictionary with annotation and checkpoint paths.
        files (dict): Dictionary with pipeline and config paths. 
        checkpt (str): string with chepoint to load for CNN.

        """
        self.paths = paths
        self.files = files
        self.checkpt = checkpt

    def import_gestures_lib(self):
        """
        Imports the hand detection dependencies.

        """
        from cvzone.HandTrackingModule import HandDetector
        self.HandDetector = HandDetector

    def gripper_gesture_control(self, rc, detector, cap, show = False):
        """
        Function used to control the gripper with hand gestures.

        Parameters:
        rc (object): RobotControl object for program execution.
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
                        rc.Gripper_State.set_value(ua.DataValue(False))
                        time.sleep(0.1)
                    if length >=100.0:
                        rc.Gripper_State.set_value(ua.DataValue(True))
                        time.sleep(0.1)
        else:
            img = np.zeros_like(img)

        cv2.imshow("Gestures", img)

    def main_packet_detect(self, rc):
            """
            Basic main packet detection.

            Parameters:
            rc (object): RobotControl object for program execution.
        
            Returns:
            tuple(np.ndarray, list): Image  with detections and detections.
        
            """
            rc.show_boot_screen('STARTING NEURAL NET...')
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
            apriltag.load_world_points('conveyor_points.json')
            pack_detect = PacketDetector(self.paths, self.files, self.checkpt)
            homography = None
            while True:
                start_time = time.time()
                ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
                
                rgb_frame = rgb_frame[:,240:1680]
                # rgb_frame = cv2.resize(rgb_frame, (640,480))
                height, width, depth = rgb_frame.shape
                
                # Update homography
                if frame_count == 1:
                    apriltag.detect_tags(rgb_frame)
                    homography = apriltag.compute_homog()

                rgb_frame = apriltag.draw_tags(rgb_frame)

                # If homography has been detected
                if isinstance(homography, np.ndarray):
                    # Increase counter for homography update
                    frame_count += 1
                    if frame_count >= 500:
                        frame_count = 1

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
                rc.objects_update(objects, img_detect)

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
                if (key == 27) and len(detected)> 0:
                    # cv2.destroyAllWindows()
                    break
            print(detected)
            return added_image , detected

    def main_robot_control_demo(self, rc):
        """
        Pick and place with static conveyor and hand gestures.

        Parameters:
        rc (object): RobotControl object for program execution.

        """
        self.import_gestures_lib()
        detected_img, detected = self.main_packet_detect(rc)

        rc.connect_OPCUA_server()

        world_centroid = detected[0][2]
        packet_x = round(world_centroid[0] * 10.0, 2)
        packet_y = round(world_centroid[1] * 10.0, 2)
        angle = detected[0][3]
        gripper_rot = rc.compute_gripper_rot(angle)
        packet_type = detected[0][4]

        rc.get_nodes()
        rc.Pick_Place_Select.set_value(ua.DataValue(False))
        time.sleep(0.5)
        frame_num = -1
        bpressed = 0
        dc = DepthCamera()
        gripper_ON = rc.Gripper_State.get_value()
        cap = cv2.VideoCapture(2)
        detector = self.HandDetector(detectionCon=0.8, maxHands=2)
        show_gestures = True
        while True:
            start = rc.Start_Prog.get_value()
            rob_stopped = rc.Rob_Stopped.get_value()
            abort = rc.Abort_Prog.get_value()
            encoder_vel = rc.Encoder_Vel.get_value()
            encoder_pos = rc.Encoder_Pos.get_value()

            x_pos = rc.Act_Pos_X.get_value()
            y_pos = rc.Act_Pos_Y.get_value()
            z_pos = rc.Act_Pos_Z.get_value()
            a_pos = rc.Act_Pos_A.get_value()
            b_pos = rc.Act_Pos_B.get_value()
            c_pos =rc.Act_Pos_C.get_value()
            status_pos = rc.Act_Pos_Status.get_value()
            turn_pos = rc.Act_Pos_Turn.get_value()

            prePick_done = rc.PrePick_Done.get_value()
            place_done = rc.Place_Done.get_value()

            ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
            rgb_frame = rgb_frame[:,240:1680]
            frame_num += 1
            height, width, depth = rgb_frame.shape
            # rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.2, beta=10)
            
            self.gripper_gesture_control(rc, detector, cap, show = show_gestures)
            
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
                rc.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',abort)
                rc.Abort_Prog.set_value(ua.DataValue(False))
                cap.release()
                rc.client.disconnect()
                # self.gripper_gesture_control(detector, cap, show = False)
                cv2.destroyAllWindows()
                time.sleep(0.5)
                break

            if rob_stopped:
                if key == ord('b'):
                    bpressed += 1
                    if bpressed == 5:
                        rc.change_trajectory(packet_x, 
                                                packet_y, 
                                                gripper_rot, 
                                                packet_type)
                        rc.start_program()
                        bpressed = 0
                elif key != ord('b'):
                    bpressed = 0

            if key == ord('o'):
                rc.Gripper_State.set_value(ua.DataValue(False))
                time.sleep(0.1)

            if key == ord('i'):
                rc.Gripper_State.set_value(ua.DataValue(True))
                time.sleep(0.1)
            
            if key == ord('g'):
                show_gestures = not show_gestures
                
            if key == ord('a'):
                rc.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',abort)
                time.sleep(0.5)
                
    def main_pick_place(self, rc, paths, files, check_point, info_dict):
        """
        Pick and place with static conveyor and multithreading.

        Parameters:
        rc (object): RobotControl object for program execution.
        info_pipe (object): Queue object containing data from the PLC server.

        """
        apriltag = ProcessingApriltag()
        apriltag.load_world_points('conveyor_points.json')
        ct = CentroidTracker()    
        dc = DepthCamera()    
        rc.show_boot_screen('STARTING NEURAL NET...')
        pack_detect = PacketDetector(paths, files, check_point)

        rc.connect_OPCUA_server()
        rc.get_nodes()
        rc.Pick_Place_Select.set_value(ua.DataValue(False))
        time.sleep(0.5)
        warn_count = 0
        track_frame = 0
        is_detect = False
        conv_left = False
        conv_right = False
        bbox = True
        depth_map = True
        f_data = False
        homography = None
        frame_count = 1
        while True:
            # Start timer for FPS estimation
            start_time = time.time()

            # Read data dict from PLC server
            try:
                rob_stopped = info_dict['rob_stopped']
                stop_active = info_dict['stop_active']
                prog_done = info_dict['prog_done']
            except:
                continue

            ret, depth_frame, rgb_frame, colorized_depth = dc.get_frame()
            
            rgb_frame = rgb_frame[:,240:1680]
            height, width, depth = rgb_frame.shape
            
            # Update homography
            if frame_count == 1:
                apriltag.detect_tags(rgb_frame)
                homography = apriltag.compute_homog()

            rgb_frame = apriltag.draw_tags(rgb_frame)

            # If homography has been detected
            if isinstance(homography, np.ndarray):
                # Increase counter for homography update
                frame_count += 1
                if frame_count >= 500:
                    frame_count = 1
            
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
            rc.objects_update(objects, img_detect)
            
            if depth_map:
                img_detect = cv2.addWeighted(img_detect, 0.8, heatmap, 0.3, 0)

            if f_data:
                cv2.putText(img_detect,str(info_dict),(10,25),
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
                        gripper_rot = rc.compute_gripper_rot(angle)
                        packet_type = detected[0][4]
                        rc.change_trajectory(packet_x, 
                                                packet_y, 
                                                gripper_rot, 
                                                packet_type)
                        rc.start_program()
                        bpressed = 0
                elif key != ord('b'):
                    bpressed = 0

            if key == ord('o'):
                rc.Gripper_State.set_value(ua.DataValue(False))
                time.sleep(0.1)

            if key == ord('i'):
                rc.Gripper_State.set_value(ua.DataValue(True))
                time.sleep(0.1)

            if key == ord('m') :
                conv_right = not conv_right
                rc.Conveyor_Right.set_value(ua.DataValue(conv_right))
                time.sleep(0.1)
            
            if key == ord('n'):
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
                print('Program Aborted: ',info_dict['abort'])
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
                print('Program Aborted: ',info_dict['abort'])
                rc.Abort_Prog.set_value(ua.DataValue(False))
                rc.client.disconnect()
                cv2.destroyAllWindows()
                print('[INFO]: Client disconnected.')
                time.sleep(0.5)
                break

    def main_pick_place_conveyor_w_point_cloud(self, rc, paths, files, check_point, info_dict):
        """
        Thread for pick and place with moving conveyor and point cloud operations.
        
        Parameters:
        rc (object): RobotControl object for program execution.
        info_pipe (object): Queue object containing data from the PLC server.
        
        """
        # Inititalize objects.
        apriltag = ProcessingApriltag()
        apriltag.load_world_points('conveyor_points.json')
        pt = PacketTracker(maxDisappeared=10, guard=50)    
        dc = DepthCamera()
        rc.show_boot_screen('STARTING NEURAL NET...')
        pack_detect = PacketDetector(paths, files, check_point)

        rc.connect_OPCUA_server()
        rc.get_nodes()
        rc.Laser_Enable.set_value(ua.DataValue(True))
        rc.Pick_Place_Select.set_value(ua.DataValue(False))
        time.sleep(0.5)
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

            # Read data dict from PLC server
            try:
                rob_stopped = info_dict['rob_stopped']
                stop_active = info_dict['stop_active']
                prog_done = info_dict['prog_done']
                encoder_vel = info_dict['encoder_vel']
                encoder_pos = info_dict['encoder_pos']
            except:
                continue

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
            
            # Update homography
            if frame_count == 1:
                apriltag.detect_tags(rgb_frame)
                homography = apriltag.compute_homog()

            rgb_frame = apriltag.draw_tags(rgb_frame)

            # If homography has been detected
            if isinstance(homography, np.ndarray):
                # Increase counter for homography update
                frame_count += 1
                if frame_count >= 500:
                    frame_count = 1

            # Detect packets using neural network.
            img_detect, detected = pack_detect.deep_pack_obj_detector(
                                                                rgb_frame, 
                                                                depth_frame,
                                                                encoder_pos, 
                                                                bnd_box = bbox)
            # Update tracked packets for current frame.
            registered_packets, deregistered_packets = pt.update(detected, depth_frame)
            print({
                'packs': registered_packets,
                'rob_stop': rob_stopped, 
                'stop_acti': stop_active, 
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
                world_x, world_y, dist_to_pack, packet = rc.single_pack_tracking_update(
                                                            registered_packets, 
                                                            img_detect, 
                                                            homography, 
                                                            is_detect,
                                                            x_fixed, 
                                                            track_frame,
                                                            frames_lim,
                                                            encoder_pos)
                track_result = (world_x, world_y, dist_to_pack)                                            
                #Trigger start of the pick and place program.
                rc.single_pack_tracking_program_start(
                                            track_result, 
                                            packet, 
                                            encoder_pos, 
                                            encoder_vel, 
                                            is_rob_ready, 
                                            pack_x_offsets, 
                                            pack_depths)

            # Show point cloud visualization when packets are deregistered.
            if len(deregistered_packets) > 0:
                pclv = PointCloudViz(".", deregistered_packets[-1])
                pclv.show_point_cloud()
                del pclv

            # Show depth frame overlay.
            if depth_map:
                img_detect = cv2.addWeighted(img_detect, 0.8, heatmap, 0.3, 0)

            # Show robot position data and FPS.
            if f_data:
                cv2.putText(img_detect,str(info_dict),(10,25),
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
                conv_right = not conv_right
                rc.change_conveyor_right(conv_right)
            
            if key == ord('n'):
                conv_left = not conv_left
                rc.change_conveyor_left(conv_left)

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