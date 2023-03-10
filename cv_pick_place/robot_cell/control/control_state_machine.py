import multiprocessing
import multiprocessing.connection
import multiprocessing.managers

import numpy as np
import cv2

from robot_cell.control.robot_control import RcData
from robot_cell.control.robot_control import RcCommand
from robot_cell.packet.packet_object import Packet
from robot_cell.packet.grip_position_estimation import GripPositionEstimation
from robot_cell.graphics_functions import drawText
from robot_cell.graphics_functions import colorizeDepthFrame


class RobotStateMachine:
    """
    State machine for robot control.
    """

    def __init__(
        self,
        control_pipe: multiprocessing.connection.PipeConnection,
        gripper_pose_estimator: GripPositionEstimation,
        encoder_pos_m: multiprocessing.managers.ValueProxy,
        home_xyz_coords: np.array,
        constants: dict,
        verbose: bool = False,
    ) -> None:
        """
        RobotStateMachine object constructor.

        Args:
            control_pipe (multiprocessing.connection.PipeConnection): Multiprocessing pipe object for sending commands to RobotControl object process.
            gripper_pose_estimator (GripPositionEstimation): Object for estimating best gripping positions of detected objects.
            encoder_pos_m (multiprocessing.managers.ValueProxy): Value object from multiprocessing Manager for reading encoder value from another process.
            home_xyz_coords (np.array): Numpy array of home position coordinates.
            constants (dict): Dictionary of constants defining behaviour of the state machine.
            verbose (bool): If extra information should be printed to the console.
        """

        # Init variables
        self.state = "READY"
        self.pick_list = []
        self.prepick_xyz_coords = []
        self.is_in_home_pos = False
        self.packet_to_pick = None
        self.trajectory_dict = {}
        self.previous_packet_type = 0

        # Init variables from inputs
        self.cp = control_pipe
        self.gpe = gripper_pose_estimator
        self.enc_pos = encoder_pos_m
        self.home_xyz_coords = home_xyz_coords
        self.verbose = verbose
        self.constants = constants

    def _add_to_pick_list(
        self, registered_packets: list[Packet], encoder_vel: float
    ) -> None:
        """
        Add packets which have been tracked for frame_limit frames to the pick list.

        Args:
            registered_packets (list[Packet]): List of tracked packet objects.
            encoder_vel (float): Encoder velocity.
        """

        # When speed of conveyor more than -100 it is moving to the left
        is_conv_mov = encoder_vel < -100.0

        # If at least one packet is being tracked
        if registered_packets:
            # If the conveyor is moving to the left direction
            if is_conv_mov:
                # Increase counter of frames with detections
                for packet in registered_packets:
                    packet.track_frame += 1
                    # If number of frames the packet is tracked for is larger than limit, and packet is not already in pick list
                    if (
                        packet.track_frame > self.constants["frame_limit"]
                        and not packet.in_pick_list
                    ):
                        print(
                            "[INFO]: Add packet ID: {} to pick list".format(
                                str(packet.id)
                            )
                        )
                        # Add packet to pick list
                        packet.in_pick_list = True
                        self.pick_list.append(packet)

    def _prep_pick_list(self) -> list[int]:
        """
        Prepare the list for choosing a packet by updating packet positions
        and removing packets which are too far.

        Returns:
            pick_list_positions (list[int]): List of current positions of packets.
        """

        # Read and check encoder position
        encoder_pos = self.enc_pos.value
        if encoder_pos is None:
            return []

        # Update pick list to current positions
        for packet in self.pick_list:
            # TODO: Check why is this necessary
            x, y = packet.get_centroid_from_encoder_in_px(encoder_pos)
            packet.set_centroid(x, y)
            packet.update_camera_centroid_from_encoder(encoder_pos)

        # Get list of current world x coordinates
        pick_list_positions = np.array(
            [packet.get_centroid_in_mm().x for packet in self.pick_list]
        )

        # If item is too far remove it from list
        is_valid_position = (
            pick_list_positions
            < self.constants["max_pick_distance"]
            - self.constants["grip_time_offset"]
            - 1.5 * self.constants["x_pick_offset"]
        )
        self.pick_list = np.ndarray.tolist(
            np.asanyarray(self.pick_list)[is_valid_position]
        )
        pick_list_positions = pick_list_positions[is_valid_position]

        return pick_list_positions

    def _offset_packet_depth_by_x(self, pick_pos_x: int, packet_z: int) -> float:
        """
        Change the z position for picking based on the position on the belt, because the conveyor belt is tilted.

        Args:
            pick_pos_x (int): X coordinate for picking in millimeters.
            packet_z (int): Calculated distance of packet from camera.

        Returns:
            offset_z (float): Offset z position.
        """

        offset = 6e-6 * (pick_pos_x**2) - 0.0107 * pick_pos_x + 4.2933
        offset_z = packet_z + offset
        return offset_z

    def _draw_depth_map(
        self,
        packet: Packet,
        depth: float,
        pick_point: tuple[float, float],
        shift_x,
        shift_y,
    ) -> None:
        """
        Draw depth map, position and depth of the grip used for grip estimation.

        Args:
            packet (Packet): Used packet object.
            depth (float): Pick depth.
            pick_point (tuple[float, float]): Relative pick position in image.
        """

        if packet.avg_depth_crop is not None:
            image_frame = colorizeDepthFrame(packet.avg_depth_crop)
            img_height, img_width, frame_channel_count = image_frame.shape
            text_size = img_height / 700

            dx, dy, z = pick_point
            pick_point = int(dx * img_width), int(dy * img_height)

            cv2.drawMarker(
                image_frame,
                (int((img_width // 2) + shift_x), int((img_height // 2) + shift_y)),
                (0, 0, 0),
                cv2.MARKER_CROSS,
                10,
                cv2.LINE_4,
            )

            cv2.drawMarker(
                image_frame, pick_point, (0, 0, 0), cv2.MARKER_CROSS, 10, cv2.LINE_4
            )

            # Draw packet depth value in milimeters
            text_centroid = "Z: {:.2f} (mm)".format(depth)
            drawText(
                image_frame,
                text_centroid,
                (pick_point[0] - 30, pick_point[1] + 20),
                text_size,
            )
            image_frame = cv2.resize(image_frame, (650, 650))
            cv2.imshow("Pick Pos", image_frame)

    def _get_pick_positions(self, packet_to_pick: Packet) -> dict:
        """
        Get dictionary of parameters for changing trajectory.

        Args:
            packet_to_pick (Packet): Packet choosen for picking.

        Returns:
            trajectory_dict (dict): Dictionary of parameters for changing trajectory .
        """

        print(
            f"[INFO]: Camera point:\n\tX: {packet_to_pick.camera_centroid_x:.2f}\n\tY: {packet_to_pick.camera_centroid_y:.2f}"
        )
        print(
            f"[INFO]: Homography point:\n\tX: {packet_to_pick.get_centroid_in_mm().x:.2f}\n\tY: {packet_to_pick.get_centroid_in_mm().y:.2f}"
        )

        if (
            packet_to_pick.camera_centroid_x is not None
            and packet_to_pick.camera_centroid_y is not None
            and False
        ):
            packet_x = packet_to_pick.camera_centroid_x
            pick_pos_y = packet_to_pick.camera_centroid_y
            print("[INFO]: Changed packet pick coordinates for camera computed ones")
        else:
            packet_x, pick_pos_y = packet_to_pick.get_centroid_in_mm()

        # Set positions and Start robot
        pick_pos_x = packet_x + self.constants["grip_time_offset"]

        angle = packet_to_pick.avg_angle_deg
        packet_type = packet_to_pick.type

        # Set packet depth to fixed value by type
        # Prediction of position by the gripper pose estimation
        # Limiting the height for packet pick positions
        z_lims = (
            self.constants["packet_depths"][packet_to_pick.type],
            self.constants["max_z"],
        )
        packet_coords = (pick_pos_x, pick_pos_y)
        y_lims = (self.constants["min_y"], self.constants["max_y"])
        (
            shift_x,
            shift_y,
            pick_pos_z,
            roll,
            pitch,
            yaw,
            pick_point,
        ) = self.gpe.estimate_from_packet(packet_to_pick, z_lims, y_lims, packet_coords)

        if packet_to_pick.camera_centroid_z is not None:
            pick_pos_z = packet_to_pick.camera_centroid_z

        if shift_x is not None:
            print(
                f"[INFO]: Estimated optimal point:\n\tZ position: {pick_pos_z:.2f}\n\tRPY angles: {roll:.2f}, {pitch:.2f}, {yaw:.2f}"
            )
            # NOTE: Pick position is always centroid for now, position estimation pick offsets are ignored
            # print(
            #     f"[INFO]: Estimated optimal point:\n\tx, y shifts: {shift_x:.2f}, {shift_y:.2f},\
            #         \n\tz position: {pick_pos_z:.2f}\n\tRPY angles: {roll:.2f}, {pitch:.2f}, {yaw:.2f}"
            # )
            # pick_pos_x += shift_x
            # pick_pos_y += shift_y
        else:
            # No pick position has been found, skip packet
            return None

        # Check if x is range
        pick_pos_x = np.clip(
            pick_pos_x,
            self.constants["min_pick_distance"],
            self.constants["max_pick_distance"] - 1.5 * self.constants["x_pick_offset"],
        )
        # Check if y is range of conveyor width and adjust accordingly
        pick_pos_y = np.clip(pick_pos_y, 75.0, 470.0)
        # Offset pick height by position on belt
        pick_pos_z = self._offset_packet_depth_by_x(pick_pos_x, pick_pos_z)
        pick_pos_z -= 5
        if pick_pos_z < 8:
            pick_pos_z = 8

        pick_pos_z = 5

        # self._draw_depth_map(packet_to_pick, pick_pos_z, pick_point, shift_x, shift_y)
        # Change end points of robot
        trajectory_dict = {
            "x": pick_pos_x,
            "y": pick_pos_y,
            "z": pick_pos_z,
            "a": roll,
            "b": pitch,
            "c": yaw,
            "x_offset": self.constants["x_pick_offset"],
            "z_offset": self.constants["z_offset"],
            "packet_type": packet_type,
            "previous_packet_type": self.previous_packet_type,
            "packet_angle": angle,
        }

        return trajectory_dict

    def _start_program(self, pick_list_positions: np.ndarray) -> tuple[Packet, dict]:
        """
        Choose a packet from pick list, set trajectory and start program.

        Args:
            pick_list_positions (np.ndarray): List of current x positions for items in pick list.

        Returns:
            packet_to_pick (Packet): Packet chosen for picking.
            trajectory_dict (dict): Dictionary of parameters for changing trajectory.
        """

        # Chose farthest item on belt
        pick_ID = pick_list_positions.argmax()
        packet_to_pick = self.pick_list.pop(pick_ID)
        print("[INFO]: Chose packet ID: {} to pick".format(str(packet_to_pick.id)))

        trajectory_dict = self._get_pick_positions(packet_to_pick)
        if trajectory_dict:
            # Set trajectory
            self.cp.send(RcData(RcCommand.CHANGE_TRAJECTORY, trajectory_dict))
            # Start robot program
            self.cp.send(RcData(RcCommand.START_PROGRAM, True))
            self.previous_packet_type = trajectory_dict["packet_type"]

        return packet_to_pick, trajectory_dict

    def _is_rob_in_pos(self, rob_pos: np.array, desired_pos: np.array) -> bool:
        """
        Check if robot is in desired position.

        Args:
            rob_pos (np.array[6]): Array of current robot positions.
            desired_pos (np.array[3]): Array of desired x, y, z position.

        Returns:
            is_in_pos (bool): True if robot is in position
        """
        curr_xyz_coords = np.array(rob_pos[0:3])  # Get x, y, z coordinates
        robot_dist = np.linalg.norm(desired_pos - curr_xyz_coords)
        is_in_pos = robot_dist < 3
        return is_in_pos

    def run(
        self,
        homography: np.ndarray,
        is_rob_ready: bool,
        registered_packets: list[Packet],
        encoder_vel: float,
        robot_interrupted: bool,
        safe_operational_stop: bool,
    ) -> str:
        """
        Run one iteration of the state machine.

        Args:
            homography (np.ndarray): Homography matrix.
            is_rob_ready (bool): Indication if robot is ready to start.
            registered_packets (list[Packet]): List of tracked packet objects.
            encoder_vel (float): Encoder velocity.

        Returns:
            state (str): Current state.
        """

        self._add_to_pick_list(registered_packets, encoder_vel)

        # Robot is ready to recieve commands
        if self.state == "READY" and is_rob_ready and homography is not None:
            pick_list_positions = self._prep_pick_list()
            # Choose a item for picking
            if (
                self.pick_list
                and pick_list_positions.max() > self.constants["min_pick_distance"]
            ):
                # Select packet and start pick place opration
                self.packet_to_pick, self.trajectory_dict = self._start_program(
                    pick_list_positions
                )
                if self.trajectory_dict:
                    # Save prepick position for use in WAIT_FOR_PACKET state
                    self.prepick_xyz_coords = np.array(
                        [
                            self.trajectory_dict["x"],
                            self.trajectory_dict["y"],
                            self.trajectory_dict["z"] + self.constants["z_offset"],
                        ]
                    )
                    self.state = "WAIT_FOR_PACKET"
                    if self.verbose:
                        print("[INFO]: State: WAIT_FOR_PACKET")
            # Send robot to home position if it isn't home already
            # elif not self.is_in_home_pos:
            #     self.cp.send(RcData(RcCommand.GO_TO_HOME))
            #     self.state = "TO_HOME_POS"
            #     if self.verbose:
            #         print("[INFO]: State: TO_HOME_POS")

        # Waiting for packet
        if self.state == "WAIT_FOR_PACKET":
            encoder_pos = self.enc_pos.value
            # Check encoder and activate robot
            # TODO: Use the new function get_centroid_from_encoder_in_mm() here
            x, y = self.packet_to_pick.get_centroid_from_encoder_in_px(encoder_pos)
            self.packet_to_pick.set_centroid(x, y)
            packet_pos_x = self.packet_to_pick.get_centroid_in_mm().x
            # print(f"Homography centroid: {packet_pos_x}")
            if self.packet_to_pick.camera_base_centroid_x is not None:
                self.packet_to_pick.update_camera_centroid_from_encoder(encoder_pos)
                packet_pos_x = self.packet_to_pick.camera_centroid_x
                # print(f"Camera centroid: {packet_pos_x}")
            # If packet is too far abort and return to ready
            if (
                packet_pos_x
                > self.trajectory_dict["x"] + self.constants["x_pick_offset"]
                and robot_interrupted
            ):
                self.cp.send(RcData(RcCommand.CONTINUE_PROGRAM))
                self.cp.send(RcData(RcCommand.GRIPPER, False))
                self.state = "READY"
                if self.verbose:
                    print("[INFO]: Missed packet, State: READY")
            # If packet is close enough continue picking operation
            elif (
                packet_pos_x
                > self.trajectory_dict["x"] - self.constants["pick_start_x_offset"]
                and robot_interrupted
            ):
                self.cp.send(RcData(RcCommand.CONTINUE_PROGRAM))
                self.state = "PLACING"
                if self.verbose:
                    print("[INFO]: State: PLACING")

            if safe_operational_stop:
                self.state = "READY"
                if self.verbose:
                    print(
                        "[WARNING]: Unable to start program in the PLC due to Operational Stop"
                    )

        # Placing packet
        if self.state == "PLACING":
            if is_rob_ready:
                self.state = "READY"
                if self.verbose:
                    print("[INFO]: State: READY")

        return self.state
