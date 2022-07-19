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
from collections import OrderedDict


class RobotCommunication:
    def __init__(self):
        """
        RobotCommunication object constructor.

        """
        
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
        self.Pick_Place_Select = self.client.get_node(
            'ns=3;s="pick_place_select"')
        self.Mult_packets = self.client.get_node(
            'ns=3;s="mult_packets"')

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

        self.ShPrePick_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."X"')
        self.ShPrePick_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."Y"')
        self.ShPrePick_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."Z"')
        self.ShPrePick_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."A"')
        self.ShPrePick_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."B"')
        self.ShPrePick_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."C"')
        self.ShPrePick_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."Status"')
        self.ShPrePick_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."Turn"')

        self.ShPick_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."X"')
        self.ShPick_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."Y"')
        self.ShPick_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."Z"')
        self.ShPick_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."A"')
        self.ShPick_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."B"')
        self.ShPick_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."C"')
        self.ShPick_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."Status"')
        self.ShPick_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."Turn"')

        self.ShPlace_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."X"')
        self.ShPlace_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Y"')
        self.ShPlace_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Z"')
        self.ShPlace_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."A"')
        self.ShPlace_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."B"')
        self.ShPlace_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."C"')
        self.ShPlace_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Status"')
        self.ShPlace_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Turn"')

        self.PrePick_Done =  self.client.get_node(
            'ns=3;s="InstPickPlace"."instPrePickPos"."Done"')
        self.Place_Done =  self.client.get_node(
            'ns=3;s="InstPickPlace"."instPlacePos"."Done"')
    
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

    def robot_server(self, info_dict):
        """
        Thread to get values from PLC server.

        Parameters:
        pipe (multiprocessing.Pipe): Sends data to another thread

        """
        # Connect server and get nodes
        self.connect_OPCUA_server()
        self.get_nodes()
        time.sleep(0.5)
        while True:
            try:
                info_dict['pos'] = self.get_actual_pos()
                info_dict['encoder_vel'] = round(self.Encoder_Vel.get_value(),2)
                info_dict['encoder_pos'] = round(self.Encoder_Pos.get_value(),2)
                info_dict['start'] = self.Start_Prog.get_value()
                info_dict['abort'] = self.Abort_Prog.get_value()
                info_dict['rob_stopped'] = self.Rob_Stopped.get_value()
                info_dict['stop_active'] = self.Stop_Active.get_value()
                info_dict['prog_done'] = self.Prog_Done.get_value()
            except:
                # Triggered when OPCUA server was disconnected
                print('[INFO]: OPCUA disconnected.')
                break

    def encoder_server(self, encoder_pos):
        """
        Thread to get encoder value from PLC server.

        Parameters:
        pipe (multiprocessing.Pipe): Sends data to another thread

        """
        # Connect server and get nodes
        self.connect_OPCUA_server()
        self.get_nodes()
        time.sleep(0.5)
        while True:
            try:
                encoder_pos.value = round(self.Encoder_Pos.get_value(), 2)
            except:
                # Triggered when OPCUA server was disconnected
                print('[INFO]: OPCUA disconnected.')
                break