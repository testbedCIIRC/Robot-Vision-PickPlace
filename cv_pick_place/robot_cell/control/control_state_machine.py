import numpy as np
from robot_cell.functions import *
from robot_cell.control.robot_control import RcData
from robot_cell.control.robot_control import RcCommand

# Default pick place constants
CONSTANTS = {
                'FRAMES_LIM' : 10, # Max frames object must be tracked to start pick & place
                'PACK_DEPTHS' : [10.0, 3.0, 5.0, 5.0], # Predefined packet depths, index corresponds to type of packet.
                'MIN_PICK_DISTANCE' : 600,  # Minimal x position in mm for packet picking
                'MAX_PICK_DISTANCE' : 1900, # Maximal x position in mm for packet picking
                'Z_OFFSET' : 50.0, # Z height offset from pick height for all positions except for pick position
                'X_PICK_OFFSET' : 140, # X offset between prepick and pick position
                'GRIP_TIME_OFFSET' : 400,  # X offset from current packet position to prepick position
                'PICK_START_X_OFFSET' : 25, # Offset between robot and packet for starting the pick move
                'MAX_Z' : 500,
                'MIN_Y' : 45.0,
                'MAX_Y' : 470.0
            }

class RobotStateMachine:
    """ State machine for robot control
    """
    def __init__(self, control_pipe, gripper_pose_estimator, encoder_pos_m, home_xyz_coords, constants = CONSTANTS, verbose = False):
            # input
            self.cp = control_pipe
            self.gpe = gripper_pose_estimator # Class for estimating gripper angles
            self.enc_pos = encoder_pos_m
            self.home_xyz_coords = home_xyz_coords
            self.verbose = verbose
            self.constants = constants
            # init
            self.state = 'READY'
            self.pick_list = []
            self.prepick_xyz_coords = []
            self.is_in_home_pos = False
            self.packet_to_pick = None
            self.trajectory_dict = {}            


    def _add_to_pick_list(self, registered_packets, encoder_vel):
        """Add packets which have been tracked for FRAMES_LIM frames to the pick list

        Args:
            pick_list (list[packets]): list of packets ready to be picked
            registered_packets (list[packet_object]): List of tracked packet objects
            encoder_vel (double): encoder velocity
        """
        # When speed of conveyor more than -100 it is moving to the left
        is_conv_mov = encoder_vel < - 100.0

        # Add to pick list
        # If packets are being tracked.
        if registered_packets:
            # If the conveyor is moving to the left direction.
            if is_conv_mov:
                # Increase counter of frames with detections.
                for packet in registered_packets:
                    packet.track_frame += 1
                    # If counter larger than limit, and packet not already in pick list.
                    if packet.track_frame > self.constants['FRAMES_LIM'] and not packet.in_pick_list:
                        print("[INFO]: Add packet ID: {} to pick list".format(str(packet.id)))
                        # Add to pick list.
                        packet.in_pick_list = True
                        self.pick_list.append(packet)


    def _prep_pick_list(self, homography):
        """
        Prepare the list for choosing a packet by updating packet positions
        and removing packets which are too far.

        Args:
            pick_list (list): List of packets ready to be picked, contains packet type objects
            homography (numpy.ndarray): Homography matrix
        Returns:
            pick_list: (list): Updated pick list
            pick_list_positions (list[int]): List of current position of packets
        """
        encoder_pos = self.enc_pos.value
        if encoder_pos is None:
            return [], []
        # Update pick list to current positions
        for packet in self.pick_list:
            packet.centroid = packet.getCentroidFromEncoder(encoder_pos)
        # Get list of current world x coordinates
        pick_list_positions = np.array([packet.getCentroidInWorldFrame(homography)[0] for packet in self.pick_list])
        # If item is too far remove it from list
        is_valid_position = pick_list_positions < self.constants['MAX_PICK_DISTANCE'] - self.constants['GRIP_TIME_OFFSET'] - 1.5*self.constants['X_PICK_OFFSET']
        self.pick_list = np.ndarray.tolist(np.asanyarray(self.pick_list)[is_valid_position])     
        pick_list_positions = pick_list_positions[is_valid_position]
        return pick_list_positions


    def _offset_packet_depth_by_x(self, pick_pos_x, packet_z):
        """
        Change the z position for picking based on the position on the belt, because the conveyor belt is tilted.

        Args:
            pick_pos_x (int): X coordinate for picking in mm
            packet_z (int): Callculated depth
        """
        offset = 6e-6*(pick_pos_x*pick_pos_x)  - 0.0107*pick_pos_x + 4.2933
        return packet_z + offset


    def _get_pick_positions(self, packet_to_pick, homography):
        """
        Get dictionary of parameters for changing trajectory

        Args:
            packet_to_pick (packet): Packet choosen for picking
            homography (numpy.ndarray): Homography matrix.
        Returns:
            trajectory_dict (dict): Dictionary of parameters for changing trajectory 
        """
        # Set positions and Start robot
        packet_x,pick_pos_y = packet_to_pick.getCentroidInWorldFrame(homography)
        pick_pos_x = packet_x + self.constants['GRIP_TIME_OFFSET']

        angle = packet_to_pick.angle
        gripper_rot = compute_gripper_rot(angle)        # TODO: Use rotation
        packet_type = packet_to_pick.type

        # Set packet depth to fixed value by type                
        # Prediction of position by the gripper pose estimation
        # Limiting the height for packet pick positions
        z_lims = (self.constants['PACK_DEPTHS'][packet_to_pick.type], self.constants['MAX_Z'])
        packet_coords = (pick_pos_x, pick_pos_y)
        y_lims = (self.constants['MIN_Y'], self.constants['MAX_Y'])
        shift_x, shift_y, pick_pos_z, roll, pitch, yaw = self.gpe.estimate_from_packet(packet_to_pick, z_lims, y_lims, packet_coords)
        if shift_x is not None:
            print(f"[INFO]: Estimated optimal point:\n\tx, y shifts: {shift_x:.2f}, {shift_y:.2f},\
                    \n\tz position: {pick_pos_z:.2f}\n\tRPY angles: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
            pick_pos_y += shift_y
        else: 
            # TODO: Implement behaviour in the future 
            # Either continue with centroid or skip packet IDK, TBD
            pass

        # Check if x is range
        pick_pos_x = np.clip(pick_pos_x, self.constants['MIN_PICK_DISTANCE'], self.constants['MAX_PICK_DISTANCE'] - 1.5*self.constants['X_PICK_OFFSET'])
        # Check if y is range of conveyor width and adjust accordingly
        pick_pos_y = np.clip(pick_pos_y, 75.0, 470.0)
        # Offset pick height by position on belt
        pick_pos_z = self._offset_packet_depth_by_x(pick_pos_x, pick_pos_z)
        

        # Change end points of robot.   
        trajectory_dict = {
            'x': pick_pos_x,
            'y': pick_pos_y,
            'rot': gripper_rot,
            'packet_type': packet_type,
            'x_offset': self.constants['X_PICK_OFFSET'],
            'pack_z': pick_pos_z,
            'a': roll,
            'b': pitch,
            'c': yaw,
            'z_offset': self.constants['Z_OFFSET'],
            'shift_x': shift_x
            }

        return trajectory_dict


    def _start_program(self, pick_list_positions, homography):
        """
        Choose a packet from pick list, set trajectory and start programm.
        
        Args:
            pick_list_positions (numpy.ndarray[float64]): List of current x positions for items in pick list
            homography (numpy.ndarray): Homography matrix.
            control_pipe (Pipe): Communication pipe between processes
        Returns:
            packet_to_pick (packet): Packet choosen for picking
            trajectory_dict (dict): Dictionary of parameters for changing trajectory 
        """
        # Chose farthest item on belt
        pick_ID = pick_list_positions.argmax()
        packet_to_pick = self.pick_list.pop(pick_ID)
        print("[INFO]: Chose packet ID: {} to pick".format(str(packet_to_pick.id)))

        trajectory_dict = self._get_pick_positions(packet_to_pick, homography)


        # Set trajectory
        self.cp.send(RcData(RcCommand.CHANGE_SHORT_TRAJECTORY, trajectory_dict))
        # Start robot program.
        self.cp.send(RcData(RcCommand.START_PROGRAM, True))

        return packet_to_pick, trajectory_dict

    def _is_rob_in_pos(self, rob_pos, desired_pos):
        """ Check if robot is in desired position
        Args:
            rob_pos (np.array[6]): Array of current robot positions
            desired_pos (np.array[3]): Array of desired x,y,z position
        Returns:
            _type_: _description_
        """
        curr_xyz_coords = np.array(rob_pos[0:3])        # get x,y,z coordinates
        robot_dist = np.linalg.norm(desired_pos-curr_xyz_coords)
        return robot_dist < 3


    def run(self, homography, is_rob_ready, registered_packets, encoder_vel, pos):
        """Run the state machine

        Args:
            homography (numpy.ndarray): Homography matrix.
            is_rob_ready (bool): indication if robot is ready to start
            registered_packets (list[packets]): _description_
            encoder_vel (double): encoder velocity
            pos (np.ndarray): current robot position
        Returns:
            state (string): current state
        """
        
        self._add_to_pick_list(registered_packets, encoder_vel) 

        # robot is ready to recieve commands
        if self.state == "READY" and is_rob_ready and homography is not None:
            pick_list_positions = self._prep_pick_list(homography)
            # Choose a item for picking
            if self.pick_list and pick_list_positions.max() > self.constants['MIN_PICK_DISTANCE']:
                # Select packet and start pick place opration
                self.packet_to_pick, self.trajectory_dict = self._start_program(pick_list_positions, homography)
                # Save prepick position for use in TO_PREPICK state
                self.prepick_xyz_coords = np.array([self.trajectory_dict['x'], self.trajectory_dict['y'], self.trajectory_dict['pack_z'] + self.constants['Z_OFFSET']])
                self.is_in_home_pos = False
                self.state = "TO_PREPICK"
                if self.verbose : print("[INFO]: state TO_PREPICK")
            # send robot to home position if it itsn't already
            elif not self.is_in_home_pos:
                self.cp.send(RcData(RcCommand.GO_TO_HOME))
                self.state = "TO_HOME_POS"
                if self.verbose : print("[INFO]: state TO_HOME_POS")

        # moving to home position
        if self.state == "TO_HOME_POS":
            if is_rob_ready and self._is_rob_in_pos(pos, self.home_xyz_coords):
                self.is_in_home_pos = True
                self.state = "READY"
                if self.verbose : print("[INFO]: state READY")

        # moving to prepick position
        if self.state == "TO_PREPICK":
            # check if robot arrived to prepick position
            if self._is_rob_in_pos(pos, self.prepick_xyz_coords):
                self.state = "WAIT_FOR_PACKET"
                if self.verbose : print("[INFO]: state WAIT_FOR_PACKET")

        # waiting for packet
        if self.state == "WAIT_FOR_PACKET":
            encoder_pos = self.enc_pos.value
            # check encoder and activate robot 
            self.packet_to_pick.centroid = self.packet_to_pick.getCentroidFromEncoder(encoder_pos)
            packet_pos_x = self.packet_to_pick.getCentroidInWorldFrame(homography)[0]
            # If packet is too far abort and return to ready
            if packet_pos_x > self.trajectory_dict['x'] + self.constants['X_PICK_OFFSET']:
                self.cp.send(RcData(RcCommand.CONTINUE_PROGRAM))
                self.cp.send(RcData(RcCommand.ABORT_PROGRAM))
                self.cp.send(RcData(RcCommand.GRIPPER, False))
                self.state = "READY"
                if self.verbose : print("[INFO]: missed packet, state READY")
            # If packet is close enough continue picking operation
            elif packet_pos_x > self.trajectory_dict['x'] - self.constants['PICK_START_X_OFFSET'] - self.trajectory_dict['shift_x']:
                self.cp.send(RcData(RcCommand.CONTINUE_PROGRAM))
                self.state = "PLACING"
                if self.verbose : print("[INFO]: state PLACING")

        # placing packet
        if self.state == "PLACING":
            if is_rob_ready:
                self.state = "READY"
                if self.verbose : print("[INFO]: state READY")

        return self.state
