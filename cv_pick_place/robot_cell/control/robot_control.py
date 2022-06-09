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

from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.packet.centroidtracker import CentroidTracker
from robot_cell.packet.packettracker import PacketTracker


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
        self.Laser_Enable = self.client.get_node(
            'ns=3;s="laser_field_enable"')

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

    def continue_program(self):
        """
        Continue robot action.

        """
        self.Conti_Prog.set_value(ua.DataValue(True))
        time.sleep(0.5)
        self.Conti_Prog.set_value(ua.DataValue(False))
        print('[INFO]: Program continued.')

    def stop_program(self):
        """
        Stop robot action.

        """
        self.Stop_Prog.set_value(ua.DataValue(True))
        print('[INFO]: Program interrupted.')
        time.sleep(0.5)
        self.Stop_Prog.set_value(ua.DataValue(False))
    def abort_program(self):
        """
        Abort robot action.

        """
        self.Abort_Prog.set_value(ua.DataValue(True))
        print('[INFO]: Program aborted.')
        time.sleep(0.5)

    def close_program(self):
        """
        Close robot program.

        """
        self.Abort_Prog.set_value(ua.DataValue(True))
        print('[INFO]: Program aborted.')
        self.Abort_Prog.set_value(ua.DataValue(False))
        self.Conti_Prog.set_value(ua.DataValue(False))
        self.client.disconnect()
        cv2.destroyAllWindows()
        print('[INFO]: Client disconnected.')
        time.sleep(0.5)

    def change_gripper_state(self, state):
        """
        Switch gripper on/off.

        """
        self.Gripper_State.set_value(ua.DataValue(state))
        print('[INFO]: Gripper state is {}.'.format(state))
        time.sleep(0.1)
    
    def change_conveyor_right(self, conv_right):
        """
        Switch conveyor right direction on/off.

        """
        conv_right = not conv_right
        self.Conveyor_Left.set_value(ua.DataValue(False))
        self.Conveyor_Right.set_value(ua.DataValue(conv_right))
        time.sleep(0.4)
        return conv_right
    
    def change_conveyor_left(self, conv_left):
        """
        Switch conveyor left direction on/off.

        """
        conv_left = not conv_left
        self.Conveyor_Right.set_value(ua.DataValue(False))
        self.Conveyor_Left.set_value(ua.DataValue(conv_left))
        time.sleep(0.4)
        return conv_left

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
        c_pos = self.Act_Pos_C.get_value()
        status_pos = self.Act_Pos_Status.get_value()
        turn_pos = self.Act_Pos_Turn.get_value()
        x_pos = round(x_pos,2)
        y_pos = round(y_pos,2)
        z_pos = round(z_pos,2)
        a_pos = round(a_pos,2)
        b_pos = round(b_pos,2)
        c_pos = round(c_pos,2)
        return x_pos, y_pos, z_pos, a_pos, b_pos, c_pos, status_pos, turn_pos  

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
        encoder_pos (float): current encoder position.
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

    def pack_obj_tracking_program_start(self, track_result, packet, encoder_pos, encoder_vel, is_rob_ready, 
                        pack_x_offsets, pack_depths ):
        """
        Triggers start of the program based on track result and robot status.
    
        Parameters:
        track_result (tuple): Updated x, mean y packet pick positions and distance to packet.
        packet (object): Final tracked packet object used for program start.
        encoder_pos (float): Current encoder position.
        encoder_vel (float): Current encoder velocity.
        is_rob_ready (bool): Boolean true if robot is ready to start program.
        pack_x_offsets (list):List of offsets for pick position.
        pack_depths (list): List of packet depths.

        """

        if track_result is not None:
            dist_to_pack = track_result[2]
            delay = dist_to_pack/(abs(encoder_vel)/10)
            delay = round(delay,2)
            if  is_rob_ready:
                packet_x = track_result[0]
                packet_y = track_result[1]
                angle = packet.angle
                gripper_rot = self.compute_gripper_rot(angle)
                packet_type = packet.pack_type
                print(packet_x, packet_y)
                self.change_trajectory(packet_x,
                                    packet_y, 
                                    gripper_rot, 
                                    packet_type,
                                    x_offset = pack_x_offsets[packet_type],
                                    pack_z = pack_depths[packet_type])
                self.Start_Prog.set_value(ua.DataValue(True))
                print('Program Started')
                time.sleep(0.5)
                self.Start_Prog.set_value(ua.DataValue(False))
                time.sleep(0.5)