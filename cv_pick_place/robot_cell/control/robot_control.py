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
from robot_cell.control.robot_communication import RobotCommunication
from enum import Enum


class RcCommand(Enum):
    GRIPPER = 1
    CONVEYOR_LEFT = 2
    CONVEYOR_RIGHT = 3
    ABORT_PROGRAM = 4
    CONTINUE_PROGRAM = 5
    STOP_PROGRAM = 6
    CLOSE_PROGRAM = 7
    START_PROGRAM = 8
    CHANGE_TRAJECTORY = 9


class RcData():
    def __init__(self, command, data = None):
        self.command = command
        self.data = data
    

class RobotControl(RobotCommunication):
    def __init__(self, rob_dict):
        """
        RobotControl object constructor.
    
        Parameters:
        rob_dict (dict): Dictionary with robot points for program. 

        """
        self.rob_dict = rob_dict

        # Inherit RobotCommunication.
        super().__init__()

    def show_boot_screen(self, message, resolution = (960,1280)):
        """
        Opens main frame window with boot screen message.

        Parameters:
        message (str): Message to be displayed.

        """
        boot_screen = np.zeros(resolution)
        cv2.namedWindow('Frame')
        cv2.putText(boot_screen, message, 
                    (resolution[0] // 2 - 150, resolution[1] // 2),
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
        
    def start_program(self, mode = False):
        """
        Start robot program.

        """
        self.Pick_Place_Select.set_value(ua.DataValue(mode))
        self.Start_Prog.set_value(ua.DataValue(True))
        print('[INFO]: Program started.')
        time.sleep(0.5)
        self.Start_Prog.set_value(ua.DataValue(False))
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
        self.Conveyor_Left.set_value(ua.DataValue(False))
        self.Conveyor_Right.set_value(ua.DataValue(conv_right))
        time.sleep(0.4)
    
    def change_conveyor_left(self, conv_left):
        """
        Switch conveyor left direction on/off.

        """
        self.Conveyor_Right.set_value(ua.DataValue(False))
        self.Conveyor_Left.set_value(ua.DataValue(conv_left))
        time.sleep(0.4)

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

    def compute_mean_packet_z(self, packet, pack_z_fixed):
        """
        Computes depth of packet based on average of stored depth frames.
    
        Parameters:

        packet (object): Final tracked packet object used for program start.

        """
        conv2cam_dist = 777.0 # mm
        # range 25 - 13
        depth_mean = np.mean(packet.depth_maps, axis = 2)
        d_rows, d_cols = depth_mean.shape

        print(d_rows, d_cols)
        
        # If depth frames are present.
        try:
            if d_rows > 0:
                # Compute centroid in depth crop coordinates.
                cx, cy = packet.centroid
                xminbbx = packet.xminbbx
                yminbbx = packet.yminbbx
                x_depth, y_depth = int(cx - xminbbx), int(cy - yminbbx)

                # Get centroid from depth mean crop.
                centroid_depth = depth_mean[y_depth, x_depth]
                print('Centroid_depth:',centroid_depth)

                # Compute packet z position with respect to conveyor base.
                pack_z = abs(conv2cam_dist - centroid_depth)

                # Return pack_z if in acceptable range, set to default if not.
                pack_z_in_range = (pack_z > pack_z_fixed) and (pack_z < pack_z_fixed + 17.0)

                if pack_z_in_range:
                    print('[Info]: Pack z in range')
                    return pack_z
                else: return pack_z_fixed

            # When depth frames unavailable.
            else: return pack_z_fixed
        
        except:
            return pack_z_fixed

    def objects_update(self, objects, image):
        """
        Draws the IDs of tracked objects.
    
        Parameters:
        objects (OrderedDict): Ordered Dictionary with currently tracked objects.
        image (np.array): Image where the objects will be drawn.

        """
        # Loop over the tracked objects.
        for (objectID, centroid) in objects.items():
            # Draw both the ID and centroid of objects.
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] , centroid[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)

    def single_pack_tracking_update(self, objects, img, homog, enable, x_fixed, 
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
        # Loop over the tracked objects.
        for (objectID, packet) in objects.items():
            # Draw both the ID and centroid of packet objects.
            centroid_tup = packet.centroid
            centroid = np.array([centroid_tup[0],centroid_tup[1]]).astype('int')
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] , centroid[1] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(img, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)
            cv2.circle(img, packet.getCentroidFromEncoder(encoder_pos), 4, (0, 0, 255), -1)

            # Compute homography if it isn't None.
            if homog is not None:
                new_centroid = np.append(centroid,1)
                world_centroid = homog.dot(new_centroid)
                world_centroid = world_centroid[0], world_centroid[1]
                cv2.putText(img, 
                            str(round(world_centroid[0],2)) +','+ 
                            str(round(world_centroid[1],2)), centroid, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # If objects are being detected.
                if enable:
                    # Append object id, and centroid id world coordinates to list.
                    track_list.append([objectID,world_centroid[0],world_centroid[1]])
                    
                    # If max number of traking frames has been reached.
                    if track_frame == frames_lim:

                        # Find the most repeated packet according to id.
                        track_array = np.array(track_list)
                        track_IDs = track_array[:,0]
                        max_ID = np.max(track_IDs)
                        track_data = track_array[track_IDs == max_ID]
                        
                        
                        # Find last recorded x pos and compute mean y.
                        last_x = float(track_data[-1,1])
                        mean_y = float(np.mean(track_data[:,2]))
                        
                        # Set world x to fixed value, convert to milimeters and round.
                        world_x = x_fixed
                        world_y = round(mean_y * 10.0,2)
                        world_last_x = round(last_x * 10.0,2)

                        # Compute distance to packet.
                        dist_to_pack = world_x - world_last_x

                        # Check if y is range of conveyor width and adjust accordingly.
                        if world_y < 75.0:
                            world_y = 75.0

                        elif world_y > 470.0:
                            world_y = 470.0
                            
                        # Empty list for tracking and reset mean variables.
                        track_list.clear()
                        last_x = 0
                        mean_y = 0
                        # Return tuple with packet to be picked data and packet object.
                        return world_x, world_y, dist_to_pack, packet

        #If max number of traking frames hasn't been reached return None.
        return None, None, None, None

    def single_pack_tracking_program_start(self, track_result, packet, encoder_pos, encoder_vel, is_rob_ready, 
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
        # If track result is available.
        if None not in track_result:
            # Compute distance to packet and delay required to continue program.
            dist_to_pack = track_result[2]
            delay = dist_to_pack/(abs(encoder_vel)/10)
            delay = round(delay,2)

            # If the robot is ready.
            if  is_rob_ready:
                # Define packet pos based on track result data.
                packet_x = track_result[0]
                packet_y = track_result[1]

                # Get gripper rotation and packet type based on last detected packet.
                angle = packet.angle
                gripper_rot = self.compute_gripper_rot(angle)
                packet_type = packet.pack_type

                # Compute packet z based on depth frame.
                pack_z_fixed = pack_depths[packet_type]
                packet_z = self.compute_mean_packet_z(packet, pack_z_fixed)
                
                print(packet_z, pack_z_fixed)
                # Change end points of robot.
                self.change_trajectory(
                                packet_x,
                                packet_y, 
                                gripper_rot, 
                                packet_type,
                                x_offset = pack_x_offsets[packet_type],
                                pack_z = packet_z)

                # Start robot program.
                self.start_program()

    def control_server(self, pipe):
        # Connect server and get nodes
        self.connect_OPCUA_server()
        self.get_nodes()

        # Enabling laser sensor to synchornize robot with moving packets.
        self.Laser_Enable.set_value(ua.DataValue(False))
        self.Pick_Place_Select.set_value(ua.DataValue(False))
        time.sleep(0.5)
        while True:
            try:
                input = pipe.recv()
                command = input.command
                data = input.data

                if command == RcCommand.GRIPPER:
                    self.change_gripper_state(data)

                elif command == RcCommand.CONVEYOR_LEFT:
                    self.change_conveyor_left(data)

                elif command == RcCommand.CONVEYOR_RIGHT:
                    self.change_conveyor_right(data)

                elif command == RcCommand.ABORT_PROGRAM:
                    self.abort_program()

                elif command == RcCommand.CONTINUE_PROGRAM:
                    self.continue_program()

                elif command == RcCommand.STOP_PROGRAM:
                    self.stop_program()

                elif command == RcCommand.CLOSE_PROGRAM:
                    self.close_program()

                elif command == RcCommand.START_PROGRAM:
                    self.start_program()

                elif command == RcCommand.CHANGE_TRAJECTORY:
                    try:
                        self.change_trajectory(data['x'], 
                                                data['y'],
                                                data['rot'],
                                                data['packet_type'],
                                                x_offset=data['x_offset'],
                                                pack_z=data['pack_z'])
                    except Exception as e:
                        print(e)
                        print("ERROR")

                else:
                    print('[WARNING]: Wrong command send to control server')


            except:
                # Triggered when OPCUA server was disconnected
                print('[INFO]: OPCUA disconnected')
                break
