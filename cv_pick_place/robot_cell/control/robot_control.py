import time
import multiprocessing
from enum import Enum
from collections import OrderedDict

import cv2
import numpy as np
from opcua import ua

from robot_cell.control.robot_communication import RobotCommunication
from robot_cell.packet.packet_object import Packet


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
    CHANGE_SHORT_TRAJECTORY = 10
    GO_TO_HOME = 11
    SET_HOME_POS_SH = 12


class RcData():
    def __init__(self, command, data = None):
        self.command = command
        self.data = data
    
class RobotControl(RobotCommunication):
    """
    Class for sending commands to robot using OPCUA. Inherits RobotCommunication.
    """

    def __init__(self, rob_dict: dict, verbose: bool = False):
        """
        RobotControl object constructor.
    
        Args:
            rob_dict (dict): Dictionary with robot points for program. 
            verbose (bool): If extra information should be printed to the console.
        """

        self.rob_dict = rob_dict
        self.verbose = verbose

        # Inherit RobotCommunication
        super().__init__()

    def show_boot_screen(self, message: str, resolution: tuple[int, int] = (540, 960)):
        """
        Opens main frame window with boot screen message.

        Args:
            message (str): Message to be displayed.
            resolution (tuple[int, int]): Resolution of the window.
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
        if self.verbose : print('[INFO]: Program continued.')

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
        self.Abort_Prog.set_value(ua.DataValue(False))
        
    def start_program(self, mode: bool = False):
        """
        Start robot program.

        Args:
            mode (bool): PLC program mode selection.
        """

        self.Pick_Place_Select.set_value(ua.DataValue(mode))
        self.Start_Prog.set_value(ua.DataValue(True))
        if self.verbose : print('[INFO]: Program started.')
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

    def change_gripper_state(self, state: bool):
        """
        Switch gripper on/off.

        Args:
            state (bool): If the gripper should be turned on or off.
        """

        self.Gripper_State.set_value(ua.DataValue(state))
        if self.verbose : print('[INFO]: Gripper state is {}.'.format(state))
        time.sleep(0.1)
    
    def change_conveyor_right(self, conv_right: bool):
        """
        Switch conveyor right direction on/off.

        Args:
            conv_right (bool): If the conveyor should be turned on or off in the right direction.
        """

        self.Conveyor_Left.set_value(ua.DataValue(False))
        self.Conveyor_Right.set_value(ua.DataValue(conv_right))
        time.sleep(0.4)
    
    def change_conveyor_left(self, conv_left: bool):
        """
        Switch conveyor left direction on/off.

        Args:
            conv_left (bool): If the conveyor should be turned on or off in the left direction.
        """

        self.Conveyor_Right.set_value(ua.DataValue(False))
        self.Conveyor_Left.set_value(ua.DataValue(conv_left))
        time.sleep(0.4)
    
    def go_to_home(self):
        """
        Send robot to home position.
        """

        self.Go_to_home.set_value(ua.DataValue(True))
        if self.verbose : print('[INFO]: Sent robot to home pos.')
        time.sleep(0.4)
        self.Go_to_home.set_value(ua.DataValue(False))
        time.sleep(0.4)

    def change_trajectory(self,
                          x: float,
                          y: float,
                          rot: float,
                          packet_type: int,
                          x_offset: float = 0.0,
                          pack_z: int = 5):
        """
        Updates the trajectory points for the robot program.
    
        Args:
            x (float): The pick x coordinate of the packet.
            y (float): The pick y coordinate of the packet.
            rot (float): The gripper pick rotation.
            packet_type (int): The detected packet class.
            x_offset (float): Robot x position offset from current packet position.
            pack_z (int): z coordinate of gripping position of packet.
        """

        nodes = []
        values = []

        nodes.append(self.Home_X)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['x'], ua.VariantType.Float)))
        nodes.append(self.Home_Y)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['y'], ua.VariantType.Float)))
        nodes.append(self.Home_Z)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['z'], ua.VariantType.Float)))
        nodes.append(self.Home_A)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['a'], ua.VariantType.Float)))
        nodes.append(self.Home_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['b'], ua.VariantType.Float)))
        nodes.append(self.Home_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['c'], ua.VariantType.Float)))
        nodes.append(self.Home_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['status'], ua.VariantType.Int16)))
        nodes.append(self.Home_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['turn'], ua.VariantType.Int16)))

        nodes.append(self.PrePick_Pos_X)
        values.append(ua.DataValue(ua.Variant(x, ua.VariantType.Float)))
        nodes.append(self.PrePick_Pos_Y)
        values.append(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        nodes.append(self.PrePick_Pos_Z)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['z'], ua.VariantType.Float)))
        nodes.append(self.PrePick_Pos_A)
        values.append(ua.DataValue(ua.Variant(rot, ua.VariantType.Float)))
        nodes.append(self.PrePick_Pos_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['b'], ua.VariantType.Float)))
        nodes.append(self.PrePick_Pos_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['c'], ua.VariantType.Float)))
        nodes.append(self.PrePick_Pos_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        nodes.append(self.PrePick_Pos_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        nodes.append(self.Pick_Pos_X)
        values.append(ua.DataValue(ua.Variant(x + x_offset, ua.VariantType.Float)))
        nodes.append(self.Pick_Pos_Y)
        values.append(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        nodes.append(self.Pick_Pos_Z)
        values.append(ua.DataValue(ua.Variant(pack_z, ua.VariantType.Float)))
        nodes.append(self.Pick_Pos_A)
        values.append(ua.DataValue(ua.Variant(rot, ua.VariantType.Float)))
        nodes.append(self.Pick_Pos_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['b'], ua.VariantType.Float)))
        nodes.append(self.Pick_Pos_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['c'], ua.VariantType.Float)))
        nodes.append(self.Pick_Pos_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        nodes.append(self.Pick_Pos_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        nodes.append(self.Place_Pos_X)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['x'], ua.VariantType.Float)))
        nodes.append(self.Place_Pos_Y)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['y'], ua.VariantType.Float)))
        nodes.append(self.Place_Pos_Z)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['z'], ua.VariantType.Float)))
        nodes.append(self.Place_Pos_A)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['a'], ua.VariantType.Float)))
        nodes.append(self.Place_Pos_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['b'], ua.VariantType.Float)))
        nodes.append(self.Place_Pos_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['c'], ua.VariantType.Float)))
        nodes.append(self.Place_Pos_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['status'], ua.VariantType.Int16)))
        nodes.append(self.Place_Pos_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['turn'], ua.VariantType.Int16)))

        self.client.set_values(nodes, values)

        time.sleep(0.7)

    def change_trajectory_short(self,
                                x: float,
                                y: float,
                                angle: float,
                                packet_type: int,
                                x_offset: float = 0.0,
                                pack_z: float = 5.0,
                                post_pick_y_offset: float = 470,
                                z_offset: float = 50.0,
                                a: float = 90.0,
                                b: float = 0.0,
                                c: float = 180.0):
        """
        Updates the trajectory points for the robot program.
    
        Args:
            x (float): The pick x coordinate of the packet.
            y (float): The pick y coordinate of the packet.
            rot (float): The gripper pick rotation.
            packet_type (int): The detected packet class.
            x_offset (float): X offset between prepick and pick position.
            pack_z (float): Pick height.
            post_pick_y_offset (float): Y position for post pick position.
            z_offset (float): Z height offset from pick height for all positions except for pick position.
            a (float): Angle in degrees.
            b (float): Angle in degrees.
            c (float): Angle in degrees.
        """

        nodes = []
        values = []
        rot = self.compute_reverse_gripper_rot(angle)

        nodes.append(self.ShPrePick_Pos_X)
        values.append(ua.DataValue(ua.Variant(x, ua.VariantType.Float)))
        nodes.append(self.ShPrePick_Pos_Y)
        values.append(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        nodes.append(self.ShPrePick_Pos_Z)
        values.append(ua.DataValue(ua.Variant(pack_z + z_offset, ua.VariantType.Float)))
        nodes.append(self.ShPrePick_Pos_A)
        values.append(ua.DataValue(ua.Variant(a, ua.VariantType.Float)))
        nodes.append(self.ShPrePick_Pos_B)
        values.append(ua.DataValue(ua.Variant(b, ua.VariantType.Float)))
        nodes.append(self.ShPrePick_Pos_C)
        values.append(ua.DataValue(ua.Variant(c, ua.VariantType.Float)))
        nodes.append(self.ShPrePick_Pos_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        nodes.append(self.ShPrePick_Pos_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        nodes.append(self.ShPick_Pos_X)
        values.append(ua.DataValue(ua.Variant(x + x_offset, ua.VariantType.Float)))
        nodes.append(self.ShPick_Pos_Y)
        values.append(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        nodes.append(self.ShPick_Pos_Z)
        values.append(ua.DataValue(ua.Variant(pack_z, ua.VariantType.Float)))
        nodes.append(self.ShPick_Pos_A)
        values.append(ua.DataValue(ua.Variant(a, ua.VariantType.Float)))
        nodes.append(self.ShPick_Pos_B)
        values.append(ua.DataValue(ua.Variant(b, ua.VariantType.Float)))
        nodes.append(self.ShPick_Pos_C)
        values.append(ua.DataValue(ua.Variant(c, ua.VariantType.Float)))
        nodes.append(self.ShPick_Pos_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        nodes.append(self.ShPick_Pos_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        nodes.append(self.ShPostPick_Pos_X)
        values.append(ua.DataValue(ua.Variant(x + 1.5*x_offset, ua.VariantType.Float)))
        nodes.append(self.ShPostPick_Pos_Y)
        values.append(ua.DataValue(ua.Variant(post_pick_y_offset, ua.VariantType.Float)))
        nodes.append(self.ShPostPick_Pos_Z)
        values.append(ua.DataValue(ua.Variant(pack_z + z_offset, ua.VariantType.Float)))
        nodes.append(self.ShPostPick_Pos_A)
        values.append(ua.DataValue(ua.Variant(a, ua.VariantType.Float)))
        nodes.append(self.ShPostPick_Pos_B)
        values.append(ua.DataValue(ua.Variant(b, ua.VariantType.Float)))
        nodes.append(self.ShPostPick_Pos_C)
        values.append(ua.DataValue(ua.Variant(c, ua.VariantType.Float)))
        nodes.append(self.ShPostPick_Pos_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        nodes.append(self.ShPostPick_Pos_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        nodes.append(self.ShPlace_Pos_X)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['x'], ua.VariantType.Float)))
        nodes.append(self.ShPlace_Pos_Y)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['y'], ua.VariantType.Float)))
        nodes.append(self.ShPlace_Pos_Z)
        values.append(ua.DataValue(ua.Variant(pack_z + z_offset, ua.VariantType.Float)))
        nodes.append(self.ShPlace_Pos_A)
        values.append(ua.DataValue(ua.Variant(rot, ua.VariantType.Float)))
        nodes.append(self.ShPlace_Pos_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['b'], ua.VariantType.Float)))
        nodes.append(self.ShPlace_Pos_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['c'], ua.VariantType.Float)))
        nodes.append(self.ShPlace_Pos_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['status'], ua.VariantType.Int16)))
        nodes.append(self.ShPlace_Pos_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['turn'], ua.VariantType.Int16)))

        self.client.set_values(nodes, values)

        time.sleep(0.2)

    def set_home_pos_short(self):
        """
        Set home position for short pick place to position in dictionary.
        """

        nodes = []
        values = []

        nodes.append(self.ShHome_Pos_X)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['x'], ua.VariantType.Float)))
        nodes.append(self.ShHome_Pos_Y)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['y'], ua.VariantType.Float)))
        nodes.append(self.ShHome_Pos_Z)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['z'], ua.VariantType.Float)))
        nodes.append(self.ShHome_Pos_A)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['a'], ua.VariantType.Float)))
        nodes.append(self.ShHome_Pos_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['b'], ua.VariantType.Float)))
        nodes.append(self.ShHome_Pos_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['c'], ua.VariantType.Float)))
        nodes.append(self.ShHome_Pos_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['status'], ua.VariantType.Int16)))
        nodes.append(self.ShHome_Pos_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['turn'], ua.VariantType.Int16)))

        self.client.set_values(nodes, values)

        time.sleep(0.7)

    def compute_gripper_rot(self, angle: float):
        """
        Computes the gripper rotation based on the detected packet angle. For rotating at picking.
    
        Args:
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

    def compute_reverse_gripper_rot(self, angle: float):
        """
        Computes the gripper rotation based on the detected packet angle. For rotating at placing.
    
        Args:
            angle (float): Detected angle of packet.
    
        Returns:
            float: Gripper rotation.
        """

        angle = abs(angle)
        if angle > 45:
            rot = angle
        if angle <= 45:
            rot = 90 + angle
        return rot

    def compute_mean_packet_z(self, packet: Packet, pack_z_fixed: float):
        """
        Computes depth of packet based on average of stored depth frames.
    
        Args:
            packet (Packet): Packet object for which centroid depth should be found.
            pack_z_fixed (float): Constant depth value to fall back to.
        """

        conv2cam_dist = 777.0 # mm
        # range 25 - 13
        depth_mean = np.mean(packet.depth_maps, axis = 2)
        d_rows, d_cols = depth_mean.shape

        print(d_rows, d_cols)
        
        # If depth frames are present
        try:
            if d_rows > 0:
                # Compute centroid in depth crop coordinates
                cx, cy = packet.centroid
                xminbbx = packet.xminbbx
                yminbbx = packet.yminbbx
                x_depth, y_depth = int(cx - xminbbx), int(cy - yminbbx)

                # Get centroid from depth mean crop
                centroid_depth = depth_mean[y_depth, x_depth]
                if self.verbose : print('Centroid_depth:',centroid_depth)

                # Compute packet z position with respect to conveyor base
                pack_z = abs(conv2cam_dist - centroid_depth)

                # Return pack_z if in acceptable range, set to default if not
                pack_z_in_range = (pack_z > pack_z_fixed) and (pack_z < pack_z_fixed + 17.0)

                if pack_z_in_range:
                    if self.verbose : print('[INFO]: Pack z in range')
                    return pack_z
                else: return pack_z_fixed

            # When depth frames unavailable
            else: return pack_z_fixed
        
        except:
            return pack_z_fixed

    def objects_update(self, objects: OrderedDict, image: np.ndarray):
        """
        Draws the IDs of tracked objects.
    
        Args:
            objects (OrderedDict): Ordered dictionary with currently tracked objects.
            image (np.ndarray): Image where the objects will be drawn.
        """

        # Loop over the tracked objects.
        for (objectID, centroid) in objects.items():
            # Draw both the ID and centroid of objects.
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] , centroid[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)

    def single_pack_tracking_update(self,
                                    objects: OrderedDict,
                                    img: np.ndarray,
                                    homog: np.ndarray,
                                    enable: bool,
                                    x_fixed: float,
                                    track_frame: int,
                                    frames_lim: int,
                                    encoder_pos: float,
                                    track_list = []) -> tuple[float, float, float, Packet]:
        """
        Computes distance to packet and updated x, mean y packet positions of tracked moving packets.
    
        Args:
            objects (OrderedDict): Ordered Dictionary with currently tracked packet objects.
            img (np.ndarray): Image where the objects will be drawn.
            homog (np.ndarray): Homography matrix.
            enable (bool): Boolean true if objects are detected. It enables appending of centroids.
            x_fixed (float): Fixed x pick position.
            track_frame (int): Frame tracking counter.
            frames_lim (int): Maximum number of frames for tracking.
            encoder_pos (float): Current encoder position.
            track_list (list): List where centroid positions are stored.

        Returns:
            tuple[float, float, float, Packet]: Updated x, mean y packet pick positions and distance to packet.
        """

        # Loop over the tracked objects
        for (objectID, packet) in objects.items():
            # Draw both the ID and centroid of packet objects
            centroid_tup = packet.centroid
            centroid = np.array([centroid_tup[0],centroid_tup[1]]).astype('int')
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] , centroid[1] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(img, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)
            cv2.circle(img, packet.getCentroidFromEncoder(encoder_pos), 4, (0, 0, 255), -1)

            # Compute homography if it isn't None
            if homog is not None:
                new_centroid = np.append(centroid,1)
                world_centroid = homog.dot(new_centroid)
                world_centroid = world_centroid[0], world_centroid[1]
                cv2.putText(img, 
                            str(round(world_centroid[0],2)) +','+ 
                            str(round(world_centroid[1],2)), centroid, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # If objects are being detected
                if enable:
                    # Append object id, and centroid id world coordinates to list
                    track_list.append([objectID,world_centroid[0],world_centroid[1]])
                    
                    # If max number of traking frames has been reached
                    if track_frame == frames_lim:

                        # Find the most repeated packet according to id
                        track_array = np.array(track_list)
                        track_IDs = track_array[:,0]
                        max_ID = np.max(track_IDs)
                        track_data = track_array[track_IDs == max_ID]
                        
                        
                        # Find last recorded x pos and compute mean y
                        last_x = float(track_data[-1,1])
                        mean_y = float(np.mean(track_data[:,2]))
                        
                        # Set world x to fixed value, convert to milimeters and round
                        world_x = x_fixed
                        world_y = round(mean_y * 10.0,2)
                        world_last_x = round(last_x * 10.0,2)

                        # Compute distance to packet
                        dist_to_pack = world_x - world_last_x

                        # Check if y is range of conveyor width and adjust accordingly
                        if world_y < 75.0:
                            world_y = 75.0

                        elif world_y > 470.0:
                            world_y = 470.0
                            
                        # Empty list for tracking and reset mean variables
                        track_list.clear()
                        last_x = 0
                        mean_y = 0
                        # Return tuple with packet to be picked data and packet object
                        return world_x, world_y, dist_to_pack, packet

        # If max number of traking frames hasn't been reached return None
        return None, None, None, None

    def single_pack_tracking_program_start(self,
                                           track_result: tuple,
                                           packet: Packet,
                                           encoder_pos: float,
                                           encoder_vel: float,
                                           is_rob_ready: bool,
                                           pack_x_offsets: list,
                                           pack_depths: list):
        """
        Triggers start of the program based on track result and robot status.
    
        Args:
            track_result (tuple): Updated x, mean y packet pick positions and distance to packet.
            packet (Packet): Final tracked packet object used for program start.
            encoder_pos (float): Current encoder position.
            encoder_vel (float): Current encoder velocity.
            is_rob_ready (bool): Boolean true if robot is ready to start program.
            pack_x_offsets (list): List of offsets for pick position.
            pack_depths (list): List of packet depths.
        """

        # If track result is available
        if None not in track_result:
            # Compute distance to packet and delay required to continue program
            dist_to_pack = track_result[2]
            delay = dist_to_pack/(abs(encoder_vel)/10)
            delay = round(delay,2)

            # If the robot is ready
            if  is_rob_ready:
                # Define packet pos based on track result data
                packet_x = track_result[0]
                packet_y = track_result[1]

                # Get gripper rotation and packet type based on last detected packet
                angle = packet.angle
                gripper_rot = self.compute_gripper_rot(angle)
                packet_type = packet.pack_type

                # Compute packet z based on depth frame
                pack_z_fixed = pack_depths[packet_type]
                packet_z = self.compute_mean_packet_z(packet, pack_z_fixed)
                
                print(packet_z, pack_z_fixed)
                # Change end points of robot
                self.change_trajectory(
                                packet_x,
                                packet_y, 
                                gripper_rot, 
                                packet_type,
                                x_offset = pack_x_offsets[packet_type],
                                pack_z = packet_z)

                # Start robot program
                self.start_program()

    def control_server(self, pipe: multiprocessing.connection.PipeConnection):
        """
        Process to set values on PLC server.
        Periodically check for input commands from main process.

        To add new command:
        1. Add the command to Enum at top of the file
        (example: NEW_COMMAND = 50)
        2. Add new if section to the while loop
        (example: elif command == RcCommand.NEW_COMMAND:)
        3. Send command from the main process, with optional data argument
        (example: control_pipe.send(RcData(RcCommand.NEW_COMMAND, data)))

        Args:
            pipe (multiprocessing.connection.PipeConnection): Sends data to another thread.
        """

        # Connect server and get nodes
        self.connect_OPCUA_server()
        self.get_nodes()

        # Enabling laser sensor to synchornize robot with moving packets
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
                    self.start_program(data)

                elif command == RcCommand.CHANGE_TRAJECTORY:
                    self.change_trajectory(data['x'], 
                                            data['y'],
                                            data['rot'],
                                            data['packet_type'],
                                            x_offset=data['x_offset'],
                                            pack_z=data['pack_z'])

                elif command == RcCommand.CHANGE_SHORT_TRAJECTORY:
                    self.change_trajectory_short(data['x'], 
                                            data['y'],
                                            data['rot'],
                                            data['packet_type'],
                                            x_offset=data['x_offset'],
                                            pack_z=data['pack_z'],
                                            a = data['a'],
                                            b = data['b'],
                                            c = data['c'],
                                            z_offset=data['z_offset'])
                
                elif command == RcCommand.SET_HOME_POS_SH:
                    self.set_home_pos_short()
            
                elif command == RcCommand.GO_TO_HOME:
                    self.go_to_home()

                else:
                    print('[WARNING]: Wrong command send to control server')

            except Exception as e:
                print('[ERROR]', e)
                print('[INFO] OPCUA disconnected')
                break
