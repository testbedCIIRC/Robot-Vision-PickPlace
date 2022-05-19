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

class RobotControl:
    def __init__(self, rob_dict, paths, files, checkpt):
        """
        RobotControl object constructor.
    
        Parameters:
        rob_dict (dict): Dictionary with robot points for program.
        paths (dict): Dictionary with annotation and checkpoint paths.
        files (dict): Dictionary with pipeline and config paths. 

        """
        self.rob_dict = rob_dict
        self.paths = paths
        self.files = files
        self.checkpt = checkpt

    def connect_OPCUA_server(self):
        """
        Connects OPC UA Client to Server on PLC.

        """
        password = "CIIRC"
        self.client = Client("opc.tcp://user:"+str(password)+"@10.35.91.101:4840/")
        self.client.connect()
        print('[INFO]: Client connected.')

    def get_nodes(self):
        """
        Using the client.get_node method, it gets nodes from OPCUA Server on PLC.

        """
        self.Start_Prog = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."start"')
        self.Conti_Prog = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."continue"')
        self.Stop_Prog = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."interrupt"')
        self.Abort_Prog = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."powerRobot"."command"."abort"')
        self.Prog_Done = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."status"."done"')
        self.Stop_Active = self.client.get_node(
            'ns=3;s="InstPickPlace"."instInterrupt"."BrakeActive"')
        self.Rob_Stopped = self.client.get_node(
            'ns=3;s="InstKukaControl"."instAutomaticExternal"."ROB_STOPPED"')
        self.Conveyor_Left = self.client.get_node(
            'ns=3;s="conveyor_left"')
        self.Conveyor_Right = self.client.get_node(
            'ns=3;s="conveyor_right"')
        self.Gripper_State = self.client.get_node(
            'ns=3;s="gripper_control"')
        self.Encoder_Vel = self.client.get_node(
            'ns=3;s="Encoder_1".ActualVelocity')
        self.Encoder_Pos = self.client.get_node(
            'ns=3;s="Encoder_1".ActualPosition')

        self.Act_Pos_X = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."X"')
        self.Act_Pos_Y = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Y"')
        self.Act_Pos_Z = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Z"')
        self.Act_Pos_A = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."A"')
        self.Act_Pos_B = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."B"')
        self.Act_Pos_C = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."C"')
        self.Act_Pos_Turn = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Turn"')
        self.Act_Pos_Status = self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Status"')

        self.Home_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."X"')
        self.Home_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."Y"')
        self.Home_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."Z"')
        self.Home_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."A"')
        self.Home_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."B"')
        self.Home_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."C"')
        self.Home_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."Status"')
        self.Home_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."Turn"')

        self.PrePick_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."X"')
        self.PrePick_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Y"')
        self.PrePick_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Z"')
        self.PrePick_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."A"')
        self.PrePick_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."B"')
        self.PrePick_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."C"')
        self.PrePick_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Status"')
        self.PrePick_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Turn"')

        self.Pick_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."X"')
        self.Pick_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Y"')
        self.Pick_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Z"')
        self.Pick_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."A"')
        self.Pick_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."B"')
        self.Pick_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."C"')
        self.Pick_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Status"')
        self.Pick_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Turn"')

        self.Place_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."X"')
        self.Place_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Y"')
        self.Place_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Z"')
        self.Place_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."A"')
        self.Place_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."B"')
        self.Place_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."C"')
        self.Place_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Status"')
        self.Place_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Turn"')

        self.PrePick_Done =  self.client.get_node(
            'ns=3;s="InstPickPlace"."instPrePickPos"."Done"')
        self.Place_Done =  self.client.get_node(
            'ns=3;s="InstPickPlace"."instPlacePos"."Done"')

    def show_boot_screen(self, message):
        """
        Opens main frame window with boot screen message.
    
        Parameters:
        message (str): Message to be displayed.

        """
        cv2.namedWindow('Frame')
        boot_screen = np.zeros((960,1280))
        cv2.putText(boot_screen, message, (1280//2 - 150, 960//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Frame", boot_screen)
        cv2.waitKey(1)

    def change_trajectory(self, x, y, rot, packet_type, x_offset = 0.0, pack_z = 5.0):
        """
        Updates the trajectory points for the robot program.
    
        Parameters:
        x (float): The pick x coordinate of the packet.
        y (float): The pick y coordinate of the packet.
        rot (float): The gripper pick rotation.
        packet_type (int): The detected packet class.

        """

        self.Home_X.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['home_pos'][0]['x'], ua.VariantType.Float)))
        self.Home_Y.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['home_pos'][0]['y'], ua.VariantType.Float)))
        self.Home_Z.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['home_pos'][0]['z'], ua.VariantType.Float)))
        self.Home_A.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['home_pos'][0]['a'], ua.VariantType.Float)))
        self.Home_B.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['home_pos'][0]['b'], ua.VariantType.Float)))
        self.Home_C.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['home_pos'][0]['c'], ua.VariantType.Float)))
        self.Home_Status.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['home_pos'][0]['status'], ua.VariantType.Int16)))
        self.Home_Turn.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['home_pos'][0]['turn'], ua.VariantType.Int16)))

        self.PrePick_Pos_X.set_value(ua.DataValue(ua.Variant(
            x, ua.VariantType.Float)))
        self.PrePick_Pos_Y.set_value(ua.DataValue(ua.Variant(
            y, ua.VariantType.Float)))
        self.PrePick_Pos_Z.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['z'], ua.VariantType.Float)))
        self.PrePick_Pos_A.set_value(ua.DataValue(ua.Variant(
            rot, ua.VariantType.Float)))
        self.PrePick_Pos_B.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['b'], ua.VariantType.Float)))
        self.PrePick_Pos_C.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['c'], ua.VariantType.Float)))
        self.PrePick_Pos_Status.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        self.PrePick_Pos_Turn.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        self.Pick_Pos_X.set_value(ua.DataValue(ua.Variant(
            x+x_offset, ua.VariantType.Float)))
        self.Pick_Pos_Y.set_value(ua.DataValue(ua.Variant(
            y, ua.VariantType.Float)))
        self.Pick_Pos_Z.set_value(ua.DataValue(ua.Variant(
            pack_z, ua.VariantType.Float)))
        self.Pick_Pos_A.set_value(ua.DataValue(ua.Variant(
            rot, ua.VariantType.Float)))
        self.Pick_Pos_B.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['b'], ua.VariantType.Float)))
        self.Pick_Pos_C.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['c'], ua.VariantType.Float)))
        self.Pick_Pos_Status.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        self.Pick_Pos_Turn.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        self.Place_Pos_X.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['place_pos'][packet_type]['x'], ua.VariantType.Float)))
        self.Place_Pos_Y.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['place_pos'][packet_type]['y'], ua.VariantType.Float)))
        self.Place_Pos_Z.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['place_pos'][packet_type]['z'], ua.VariantType.Float)))
        self.Place_Pos_A.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['place_pos'][packet_type]['a'], ua.VariantType.Float)))
        self.Place_Pos_B.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['place_pos'][packet_type]['b'], ua.VariantType.Float)))
        self.Place_Pos_C.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['place_pos'][packet_type]['c'], ua.VariantType.Float)))
        self.Place_Pos_Status.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['place_pos'][packet_type]['status'], ua.VariantType.Int16)))
        self.Place_Pos_Turn.set_value(ua.DataValue(ua.Variant(
            self.rob_dict['place_pos'][packet_type]['turn'], ua.VariantType.Int16)))
    
        time.sleep(0.7)

    def compute_gripper_rot(self, angle):
        """
        Computes the gripper rotation based on the detected packet angle.
    
        Parameters:
        angle (float): Detected angle of packet.
    
        Returns:
        float: Gripper rotation.

        """
        angle = abs(angle)
        if angle > 45:
            rot = 90 + (90 - angle)
        if angle <= 45:
            rot = 90 - angle
        return rot

    def get_actual_pos(self):
        """
        Reads the actual position of the robot TCP with respect to the base.
    
        Returns:
        tuple: Actual pos. of robot TCP: x, y, z, a, b, c as float. Status, turn as int.

        """
        x_pos = self.Act_Pos_X.get_value()
        y_pos = self.Act_Pos_Y.get_value()
        z_pos = self.Act_Pos_Z.get_value()
        a_pos = self.Act_Pos_A.get_value()
        b_pos = self.Act_Pos_B.get_value()
        c_pos =self.Act_Pos_C.get_value()
        status_pos = self.Act_Pos_Status.get_value()
        turn_pos = self.Act_Pos_Turn.get_value()
        x_pos = round(x_pos,2)
        y_pos = round(y_pos,2)
        z_pos = round(z_pos,2)
        a_pos = round(a_pos,2)
        b_pos = round(b_pos,2)
        c_pos = round(c_pos,2)
        return x_pos, y_pos, z_pos, a_pos, b_pos, c_pos, status_pos, turn_pos  

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
        

    def objects_update(self,objects,image):
        """
        Draws the IDs of tracked objects.
    
        Parameters:
        objects (OrderedDict): Ordered Dictionary with currently tracked objects.
        image (np.array): Image where the objects will be drawn.

        """
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] , centroid[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)

    def packet_tracking_update(self, objects, img, homog, enable, x_fixed, 
                                track_frame, frames_lim, track_list = []):
        """
        Computes distance to packet and updated x, mean y packet positions of tracked moving packets.
    
        Parameters:
        objects (OrderedDict): Ordered Dictionary with currently tracked objects.
        img (numpy.ndarray): Image where the objects will be drawn.
        homog (numpy.ndarray): Homography matrix.
        enable (bool): Boolean true if objects are detected. It enables appending of centroids.
        x_fixed (float): Fixed x pick position.
        track_frame (int): Frame tracking counter.
        frames_lim (int): Maximum number of frames for tracking.
        track_list (list):List where centroid positions are stored.

        Returns:
        tuple(float): Updated x, mean y packet pick positions and distance to packet.
    
        """
        # loop over the tracked objects
        for (objectID, data) in objects.items():
            centroid = data[:2]
            centroid = centroid.astype('int')
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] , centroid[1] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(img, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)
            if homog is not None:
                new_centroid = np.append(centroid,1)
                world_centroid = homog.dot(new_centroid)
                world_centroid = world_centroid[0], world_centroid[1]
                cv2.putText(img, 
                            str(round(world_centroid[0],2)) +','+ 
                            str(round(world_centroid[1],2)), centroid, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if enable:
                    track_list.append([objectID,world_centroid[0],world_centroid[1]])

                    if track_frame == frames_lim:
                        track_array = np.array(track_list)
                        track_IDs = track_array[:,0]
                        max_ID = np.max(track_IDs)
                        track_data = track_array[track_IDs == max_ID]
                        
                        mean_x = float(track_data[-1,1])
                        mean_y = float(np.mean(track_data[:,2]))
                        print(track_data[:,2],mean_y)    
                        
                        world_x = round(mean_x* 10.0,2)
                        world_y = round(mean_y* 10.0,2)
                        dist_to_pack = x_fixed - world_x
                        if world_y < 75.0:
                            world_y = 75.0

                        elif world_y > 470.0:
                            world_y = 470.0 
                        
                        track_list.clear()
                        mean_x = 0
                        mean_y = 0
                        return x_fixed, world_y, dist_to_pack
    def pack_obj_tracking_update(self, objects, img, homog, enable, x_fixed, 
                                track_frame, frames_lim, encoder_pos, track_list = []):
        """
        Computes distance to packet and updated x, mean y packet positions of tracked moving packets.
    
        Parameters:
        objects (OrderedDict): Ordered Dictionary with currently tracked packet objects.
        img (numpy.ndarray): Image where the objects will be drawn.
        homog (numpy.ndarray): Homography matrix.
        enable (bool): Boolean true if objects are detected. It enables appending of centroids.
        x_fixed (float): Fixed x pick position.
        track_frame (int): Frame tracking counter.
        frames_lim (int): Maximum number of frames for tracking.
        track_list (list):List where centroid positions are stored.

        Returns:
        tuple(float): Updated x, mean y packet pick positions and distance to packet.
    
        """
        # loop over the tracked objects
        for (objectID, packet) in objects.items():
            centroid_tup = packet.centroid
            centroid = np.array([centroid_tup[0],centroid_tup[1]]).astype('int')
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] , centroid[1] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(img, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)
            cv2.circle(img, packet.getCentroidFromEncoder(encoder_pos), 4, (0, 0, 255), -1)
            if homog is not None:
                new_centroid = np.append(centroid,1)
                world_centroid = homog.dot(new_centroid)
                world_centroid = world_centroid[0], world_centroid[1]
                cv2.putText(img, 
                            str(round(world_centroid[0],2)) +','+ 
                            str(round(world_centroid[1],2)), centroid, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if enable:
                    track_list.append([objectID,world_centroid[0],world_centroid[1]])

                    if track_frame == frames_lim:
                        track_array = np.array(track_list)
                        track_IDs = track_array[:,0]
                        max_ID = np.max(track_IDs)
                        track_data = track_array[track_IDs == max_ID]
                        
                        mean_x = float(track_data[-1,1])
                        mean_y = float(np.mean(track_data[:,2]))
                        print(track_data[:,2],mean_y)    
                        
                        world_x = round(mean_x* 10.0,2)
                        world_y = round(mean_y* 10.0,2)
                        dist_to_pack = x_fixed - world_x

                        if world_y < 75.0:
                            world_y = 75.0

                        elif world_y > 470.0:
                            world_y = 470.0 
                        
                        track_list.clear()
                        mean_x = 0
                        mean_y = 0
                        
                        return (x_fixed, world_y, dist_to_pack), packet
                    else: return None, None
    
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
        cap = cv2.VideoCapture(0)
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
                
    def main_pick_place(self,server_in):
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

            if rob_stopped:
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