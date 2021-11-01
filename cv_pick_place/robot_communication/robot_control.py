from opcua import Client
import datetime
import time
import sys
import os
import json
import cv2 
from opcua import ua
import pyrealsense2
import numpy as np
# from realsense_depth import *
import random
import matplotlib as mpl
import scipy.signal
from scipy import ndimage
from scipy.spatial import distance as dist
from collections import OrderedDict
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from cvzone.HandTrackingModule import HandDetector
from packet_detection.packet_detector import packet_detector
from cv2_apriltag.apriltag_detection import processing_apriltag
from realsense_config.realsense_depth import DepthCamera
from centroid_tracker.centroidtracker import CentroidTracker
class robot_control:
    def __init__(self, Pick_place_dict, paths, files, checkpt):
        self.Pick_place_dict = Pick_place_dict
        self.paths = paths
        self.files = files
        self.checkpt = checkpt
    def get_nodes(self):
        self.Start_Prog = self.client.get_node('ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."start"')
        self.Abort_Prog = self.client.get_node('ns=3;s="HMIKuka"."robot"."powerRobot"."command"."abort"')
        self.Rob_Stopped = self.client.get_node('ns=3;s="InstKukaControl"."instAutomaticExternal"."ROB_STOPPED"')
        self.Gripper_State = self.client.get_node('ns=3;s="gripper_control"')
        self.Encoder_Vel = self.client.get_node('ns=3;s="Encoder_1".ActualVelocity')
        self.Encoder_Pos = self.client.get_node('ns=3;s="Encoder_1".ActualPosition')

        self.Act_Pos_X = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."X"')
        self.Act_Pos_Y = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Y"')
        self.Act_Pos_Z = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Z"')
        self.Act_Pos_A = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."A"')
        self.Act_Pos_B = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."B"')
        self.Act_Pos_C = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."C"')
        self.Act_Pos_Turn = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Turn"')
        self.Act_Pos_Status = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Status"')

        self.PrePick_Pos_X = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."X"')
        self.PrePick_Pos_Y = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Y"')
        self.PrePick_Pos_Z = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Z"')
        self.PrePick_Pos_A = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."A"')
        self.PrePick_Pos_B = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."B"')
        self.PrePick_Pos_C = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."C"')
        self.PrePick_Pos_Status = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Status"')
        self.PrePick_Pos_Turn = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Turn"')

        self.Pick_Pos_X = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."X"')
        self.Pick_Pos_Y = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Y"')
        self.Pick_Pos_Z = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Z"')
        self.Pick_Pos_A = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."A"')
        self.Pick_Pos_B = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."B"')
        self.Pick_Pos_C = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."C"')
        self.Pick_Pos_Status = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Status"')
        self.Pick_Pos_Turn = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Turn"')

        self.Place_Pos_X = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."X"')
        self.Place_Pos_Y = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Y"')
        self.Place_Pos_Z = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Z"')
        self.Place_Pos_A = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."A"')
        self.Place_Pos_B = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."B"')
        self.Place_Pos_C = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."C"')
        self.Place_Pos_Status = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Status"')
        self.Place_Pos_Turn = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Turn"')

        self.PrePick_Done =  self.client.get_node('ns=3;s="InstPickPlace"."instPrePickPos"."Done"')
        self.Place_Done =  self.client.get_node('ns=3;s="InstPickPlace"."instPlacePos"."Done"')

    def show_boot_screen(self, message):
        cv2.namedWindow('Frame')
        boot_screen = np.zeros((960,1280))
        cv2.putText(boot_screen, message, (1280//2 - 150, 960//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Frame", boot_screen)
        cv2.waitKey(1)

    def connect_OPCUA_server(self):
        self.show_boot_screen('CONNECTING TO OPC UA SERVER...')
        password = "CIIRC"
        self.client = Client("opc.tcp://user:"+str(password)+"@10.35.91.101:4840/")
        self.client.connect()
        print("client connected")

    def change_trajectory(self, x, y, rot, packet_type):

        self.PrePick_Pos_X.set_value(ua.DataValue(ua.Variant(x, ua.VariantType.Float)))
        self.PrePick_Pos_Y.set_value(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        self.PrePick_Pos_Z.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['z'], ua.VariantType.Float)))
        self.PrePick_Pos_A.set_value(ua.DataValue(ua.Variant(rot, ua.VariantType.Float)))
        self.PrePick_Pos_B.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['b'], ua.VariantType.Float)))
        self.PrePick_Pos_C.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['c'], ua.VariantType.Float)))
        self.PrePick_Pos_Status.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        self.PrePick_Pos_Turn.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        self.Pick_Pos_X.set_value(ua.DataValue(ua.Variant(x, ua.VariantType.Float)))
        self.Pick_Pos_Y.set_value(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        self.Pick_Pos_Z.set_value(ua.DataValue(ua.Variant(3.0, ua.VariantType.Float)))
        self.Pick_Pos_A.set_value(ua.DataValue(ua.Variant(rot, ua.VariantType.Float)))
        self.Pick_Pos_B.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['b'], ua.VariantType.Float)))
        self.Pick_Pos_C.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['c'], ua.VariantType.Float)))
        self.Pick_Pos_Status.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        self.Pick_Pos_Turn.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        self.Place_Pos_X.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['x'], ua.VariantType.Float)))
        self.Place_Pos_Y.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['y'], ua.VariantType.Float)))
        self.Place_Pos_Z.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['z'], ua.VariantType.Float)))
        self.Place_Pos_A.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['a'], ua.VariantType.Float)))
        self.Place_Pos_B.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['b'], ua.VariantType.Float)))
        self.Place_Pos_C.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['c'], ua.VariantType.Float)))
        self.Place_Pos_Status.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['status'], ua.VariantType.Int16)))
        self.Place_Pos_Turn.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['turn'], ua.VariantType.Int16)))
    
        time.sleep(0.5)
    def compute_gripper_rot(self, angle):
        rot = 90.0 - abs(angle)
        return rot
        
    def gripper_gesture_control(self,detector, cap, show = False):
        
        if show:
            success, img = cap.read()
            hands, img = detector.findHands(img)  # with draw
                # hands = detector.findHands(img, draw=False)  # without draw
            if hands:
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"]  # List of 21 Landmark points
                bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
                centerPoint1 = hand1['center']  # center of the hand cx,cy
                handType1 = hand1["type"]  # Handtype Left or Right

                fingers1 = detector.fingersUp(hand1)

                if len(hands) == 2:
                    # Hand 2
                    hand2 = hands[1]
                    lmList2 = hand2["lmList"]  # List of 21 Landmark points
                    bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                    centerPoint2 = hand2['center']  # center of the hand cx,cy
                    handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                    fingers2 = detector.fingersUp(hand2)

                    # Find Distance between two Landmarks. Could be same hand or different hands
                    length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
                    length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
                    if length <=30.0:
                        self.Gripper_State.set_value(ua.DataValue(False))
                        time.sleep(0.1)
                    if length >=100.0:
                        self.Gripper_State.set_value(ua.DataValue(True))
                        time.sleep(0.1)
            cv2.imshow("Gestures", img)
        else:
            cv2.destroyWindow("Gestures")

    def objects_update(self,objects,image):
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] , centroid[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)
    def main_packet_detect(self):
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
        pack_detect = packet_detector(self.paths, self.files, self.checkpt)
        homography = None
        while True:
            start_time = time.time()
            ret, depth_frame, color_frame, colorized_depth = dc.get_frame()
            
            color_frame = color_frame[:,240:1680]
            # color_frame = cv2.resize(color_frame, (640,480))
            height, width, depth = color_frame.shape[0],color_frame.shape[1],color_frame.shape[2]
            
            apriltag = processing_apriltag(None, color_frame, None)
            try:
                color_frame = apriltag.detect_tags()
                homography = apriltag.compute_homog()

                is_marker_detect= type(homography).__module__ == np.__name__ or homography == None
                if is_marker_detect:
                    warn_count = 0
                    # print(homography)
                    
            except:
            #Triggered when no markers are in the frame:
                warn_count += 1
                if warn_count == 1:
                    print("[INFO]: Markers out of frame or moving.")
                pass

            # color_frame = cv2.convertScaleAbs(color_frame, alpha=a, beta=b)
            # print(a,b,d)
            
            depth_frame = depth_frame[90:400,97:507]
            depth_frame = cv2.resize(depth_frame, (width,height))

            # heatmap = cv2.applyColorMap(np.uint8(depth_frame*d), cv2.COLORMAP_TURBO)

            heatmap = colorized_depth
            heatmap = heatmap[90:400,97:507,:]
            heatmap = cv2.resize(heatmap, (width,height))
            
            img_np_detect, result, rects = pack_detect.deep_detector(color_frame, depth_frame, homography, bnd_box = bbox)
            
            objects = ct.update(rects)
            self.objects_update(objects, img_np_detect)

            cv2.circle(img_np_detect, (int(width/2),int(height/2) ), 4, (0, 0, 255), -1)
            added_image = cv2.addWeighted(img_np_detect, 0.8, heatmap, 0.3, 0)
            if f_rate:
                cv2.putText(added_image,'FPS:'+ str( 1.0 / (time.time() - start_time)),(60,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                print("FPS: ", 1.0 / (time.time() - start_time))
            
            cv2.imshow("Frame", cv2.resize(added_image, (1280,960)))
            # cv2.imshow("result", result)
            # cv2.imshow('object detection', cv2.resize(img_np_detect, (1280,960)))
            # cv2.imshow("Heatmap",cv2.resize(heatmap, (1280,960)))
            # cv2.imshow("Color", color_frame)
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
            if key== ord('b'):
                if bbox == False:
                    bbox = True
                else:
                    bbox = False
            if key== ord('f'):
                if f_rate == False:
                    f_rate = True
                else:
                    f_rate = False
            if key== 27:
                # cv2.destroyAllWindows()
                break
        print(rects)
        return added_image , rects

    def main_robot_control(self):
        detected_img, rects = self.main_packet_detect()

        self.connect_OPCUA_server()

        world_centroid = rects[0][2]
        packet_x = round(world_centroid[0] * 10.0, 2)
        packet_y = round(world_centroid[1] * 10.0, 2)
        angle = rects[0][3]
        gripper_rot = self.compute_gripper_rot(angle)
        packet_type = rects[0][4]

        self.get_nodes()

        frame_num = -1
        bpressed = 0
        dc = DepthCamera()
        gripper_ON = self.Gripper_State.get_value()
        cap = cv2.VideoCapture(1)
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        show_gestures = False
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

            # print(start, x_pos, y_pos, z_pos, a_pos, b_pos, c_pos)

            ret, depth_frame, color_frame, colorized_depth = dc.get_frame()
            color_frame = color_frame[:,240:1680]
            frame_num += 1
            height, width, depth = color_frame.shape[0],color_frame.shape[1],color_frame.shape[2]
            # color_frame = cv2.convertScaleAbs(color_frame, alpha=1.2, beta=10)
            
            self.gripper_gesture_control(detector, cap, show = show_gestures)
            
            x_pos = round(x_pos,2)
            y_pos = round(y_pos,2)
            z_pos = round(z_pos,2)
            a_pos = round(a_pos,2)
            b_pos = round(b_pos,2)
            c_pos = round(c_pos,2)
            encoder_vel = round(encoder_vel,2)
            encoder_pos = round(encoder_pos,2)

            cv2.circle(color_frame, (int(width/2),int(height/2) ), 4, (0, 0, 255), -1)
            cv2.putText(color_frame,'x:'+ str(x_pos),(60,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'y:'+ str(y_pos),(60,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'z:'+ str(z_pos),(60,70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'a:'+ str(a_pos),(60,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'b:'+ str(b_pos),(60,110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'c:'+ str(c_pos),(60,130),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'Status:'+ str(status_pos),(60,150),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'Turn:'+ str(turn_pos),(60,170),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'Enc. Speed:'+ str(encoder_vel),(60,190),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'Enc. Position:'+ str(encoder_pos),(60,210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            color_frame = cv2.addWeighted(color_frame, 0.8, detected_img, 0.4, 0)

            cv2.imshow("Frame", cv2.resize(color_frame, (1280,960)))
            cv2.imshow('Object detected', cv2.resize(detected_img, (1280,960)))

            key = cv2.waitKey(1)
            if key == 27:
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',abort)
                self.Abort_Prog.set_value(ua.DataValue(False))
                cap.release()
                self.client.disconnect()
                self.gripper_gesture_control(detector, cap, show = False)
                cv2.destroyWindow("Object detected")
                # cv2.destroyAllWindows()
                time.sleep(0.5)
                break

            if rob_stopped:
                if key == ord('b'):
                    bpressed += 1
                    if bpressed == 5:
                        self.change_trajectory(packet_x, packet_y, gripper_rot, packet_type)
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
                if show_gestures == False:
                    show_gestures = True
                else:
                    show_gestures = False

            if key == ord('a'):
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',abort)
                time.sleep(0.5)