import time
import multiprocessing
import multiprocessing.connection
from enum import Enum, unique, auto

import cv2
import numpy as np

import opcua
import opcua.ua

from robot_cell.control.robot_communication import RobotCommunication


@unique
class RcCommand(Enum):
    GRIPPER = auto()
    CONVEYOR_LEFT = auto()
    CONVEYOR_RIGHT = auto()
    CONTINUE_PROGRAM = auto()
    CLOSE_PROGRAM = auto()
    START_PROGRAM = auto()
    CHANGE_TRAJECTORY = auto()
    SET_HOME_POS = auto()
    TRACKING_TOGGLE = auto()
    SET_TRACKING_POS = auto()


class RcData:
    def __init__(self, command, data=None):
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
        cv2.namedWindow("Frame")
        cv2.putText(
            boot_screen,
            message,
            (resolution[1] // 2 - 150, resolution[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.imshow("Frame", boot_screen)
        cv2.waitKey(1)

    def continue_program(self):
        """
        Continue robot action.
        """

        self.Conti_Prog.set_value(opcua.ua.DataValue(True))
        if self.verbose:
            print("[INFO] Program continued")

    def start_program(self):
        """
        Start robot program.

        Args:
            mode (bool): PLC program mode selection.
        """

        self.Start_Prog.set_value(opcua.ua.DataValue(True))
        if self.verbose:
            print("[INFO] Program started")

    def close_program(self):
        """
        Close robot program.
        """

        self.client.disconnect()

    def change_gripper_state(self):
        """
        Switch gripper on/off.
        """

        self.Gripper_State.set_value(opcua.ua.DataValue(True))
        self.Gripper_State.set_value(opcua.ua.DataValue(False))
        if self.verbose:
            print("[INFO] Toggled gripper.")

    def change_conveyor_right(self):
        """
        Switch conveyor right direction on/off.
        """

        self.Conveyor_Right.set_value(opcua.ua.DataValue(True))
        self.Conveyor_Right.set_value(opcua.ua.DataValue(False))
        if self.verbose:
            print("[INFO] Toggled conveyor belt.")

    def change_conveyor_left(self):
        """
        Switch conveyor left direction on/off.
        """

        self.Conveyor_Left.set_value(opcua.ua.DataValue(True))
        self.Conveyor_Left.set_value(opcua.ua.DataValue(False))
        if self.verbose:
            print("[INFO] Toggled conveyor belt.")

    def set_trajectory(
        self,
        x: float,
        y: float,
        z: float,
        a: float,
        b: float,
        c: float,
        x_offset: float,
        z_offset: float,
        packet_type: int,
        previous_packet_type: int,
        packet_angle: float,
    ):
        """
        Updates the trajectory points for the robot program.

        Args:
            x (float): The pick x coordinate of the packet.
            y (float): The pick y coordinate of the packet.
            z (float): Pick height.
            a (float): Angle in degrees for the pick position.
            b (float): Angle in degrees for the pick position.
            c (float): Angle in degrees for the pick position.
            x_offset (float): X offset between prepick and pick position.
            z_offset (float): Z height offset from pick height for all positions except for pick position.
            packet_type (int): The detected packet class.
            previous_packet_type (int): Packet class of the previous trajectory.
            packet_angle (float): Detected angle of rotation of the packet in the RGB image frame.
        """

        nodes = []
        values = []

        # Create E6POS type instance for writing into struct nodes
        # In order for the opcua.ua.E6POS class to be available, client.load_type_definitions() has to be called first,
        # which downloads the struct definitions (E6POS and others) from the PLC OPCUA server
        start_E6POS = opcua.ua.E6POS()
        start_E6POS.X = self.rob_dict["place_pos"][previous_packet_type]["x"]
        start_E6POS.Y = self.rob_dict["place_pos"][previous_packet_type]["y"] - 50
        start_E6POS.Z = self.rob_dict["place_pos"][previous_packet_type]["z"]
        start_E6POS.A = a
        start_E6POS.B = self.rob_dict["place_pos"][previous_packet_type]["b"]
        start_E6POS.C = self.rob_dict["place_pos"][previous_packet_type]["c"]
        start_E6POS.Status = self.rob_dict["place_pos"][previous_packet_type]["status"]
        start_E6POS.Turn = self.rob_dict["place_pos"][previous_packet_type]["turn"]
        values.append(opcua.ua.DataValue(start_E6POS))
        nodes.append(self.Pos_Start)

        prepick_E6POS = opcua.ua.E6POS()
        prepick_E6POS.X = x
        prepick_E6POS.Y = y
        prepick_E6POS.Z = z + z_offset
        prepick_E6POS.A = a
        prepick_E6POS.B = b
        prepick_E6POS.C = c
        prepick_E6POS.Status = 2
        prepick_E6POS.Turn = 0
        values.append(opcua.ua.DataValue(prepick_E6POS))
        nodes.append(self.Pos_PrePick)

        pick_E6POS = opcua.ua.E6POS()
        pick_E6POS.X = x + x_offset
        pick_E6POS.Y = y
        pick_E6POS.Z = z
        pick_E6POS.A = a
        pick_E6POS.B = b
        pick_E6POS.C = c
        pick_E6POS.Status = 2
        pick_E6POS.Turn = 0
        values.append(opcua.ua.DataValue(pick_E6POS))
        nodes.append(self.Pos_Pick)

        postpick_E6POS = opcua.ua.E6POS()
        postpick_E6POS.X = x + 1.5 * x_offset
        postpick_E6POS.Y = y
        postpick_E6POS.Z = z + z_offset
        postpick_E6POS.A = a
        postpick_E6POS.B = b
        postpick_E6POS.C = c
        postpick_E6POS.Status = 2
        postpick_E6POS.Turn = 0
        values.append(opcua.ua.DataValue(postpick_E6POS))
        nodes.append(self.Pos_PostPick)

        place_E6POS = opcua.ua.E6POS()
        place_E6POS.X = self.rob_dict["place_pos"][packet_type]["x"]
        place_E6POS.Y = self.rob_dict["place_pos"][packet_type]["y"]
        place_E6POS.Z = self.rob_dict["place_pos"][packet_type]["z"]
        place_E6POS.A = (a + packet_angle) if packet_angle < 45 else packet_angle
        place_E6POS.B = self.rob_dict["place_pos"][packet_type]["b"]
        place_E6POS.C = self.rob_dict["place_pos"][packet_type]["c"]
        place_E6POS.Status = self.rob_dict["place_pos"][packet_type]["status"]
        place_E6POS.Turn = self.rob_dict["place_pos"][packet_type]["turn"]
        values.append(opcua.ua.DataValue(place_E6POS))
        nodes.append(self.Pos_Place)

        self.client.set_values(nodes, values)

    def set_home_pos(self):
        """
        Set home position for short pick place to position in dictionary.
        """

        start_E6POS = opcua.ua.E6POS()
        start_E6POS.X = self.rob_dict["home_pos"][0]["x"]
        start_E6POS.Y = self.rob_dict["home_pos"][0]["y"]
        start_E6POS.Z = self.rob_dict["home_pos"][0]["z"]
        start_E6POS.A = self.rob_dict["home_pos"][0]["a"]
        start_E6POS.B = self.rob_dict["home_pos"][0]["b"]
        start_E6POS.C = self.rob_dict["home_pos"][0]["c"]
        start_E6POS.Status = self.rob_dict["home_pos"][0]["status"]
        start_E6POS.Turn = self.rob_dict["home_pos"][0]["turn"]
        self.Pos_Start.set_value(opcua.ua.DataValue(start_E6POS))

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

    def toggle_tracking(self, track: bool):
        self.Tracking_Exec_Prog.set_value(opcua.ua.DataValue(track))
        if self.verbose:
            print("[INFO] Tracking program state is {}.".format(track))

    def set_tracking_position(self, x, y, z):
        track_E6POS = opcua.ua.E6POS()
        track_E6POS.X = x
        track_E6POS.Y = y
        track_E6POS.Z = z
        track_E6POS.A = 0.0
        track_E6POS.B = 0.0
        track_E6POS.C = 0.0
        track_E6POS.Status = 0
        track_E6POS.Turn = 0
        self.Tracking_Pos_Track.set_value(opcua.ua.DataValue(track_E6POS))

    def control_server(self, pipe: multiprocessing.connection.PipeConnection):
        """
        Process to set values on PLC server.
        Periodically check for input commands from main process.

        To add new command:
        1. Add the command to Enum at top of the file
        (example: NEW_COMMAND = auto())
        2. Add new if section to the while loop
        (example: elif command == RcCommand.NEW_COMMAND:)
        3. Send command from the main process, with optional data argument
        (example: control_pipe.send(RcData(RcCommand.NEW_COMMAND, data)))

        Args:
            pipe (multiprocessing.connection.PipeConnection): Sends data to another thread.
        """

        # Connect server and get nodes
        self.connect_OPCUA_server()

        if not self.connected:
            while True:
                input = pipe.recv()
                time.sleep(0.1)
                pass

        self.get_nodes()
        time.sleep(0.5)

        while True:
            try:
                input = pipe.recv()
                command = input.command
                data = input.data

                if command == RcCommand.GRIPPER:
                    self.change_gripper_state()

                elif command == RcCommand.CONVEYOR_LEFT:
                    self.change_conveyor_left()

                elif command == RcCommand.CONVEYOR_RIGHT:
                    self.change_conveyor_right()

                elif command == RcCommand.CONTINUE_PROGRAM:
                    self.continue_program()

                elif command == RcCommand.CLOSE_PROGRAM:
                    self.close_program()

                elif command == RcCommand.START_PROGRAM:
                    self.start_program()

                elif command == RcCommand.CHANGE_TRAJECTORY:
                    self.set_trajectory(
                        data["x"],
                        data["y"],
                        data["z"],
                        data["a"],
                        data["b"],
                        data["c"],
                        data["x_offset"],
                        data["z_offset"],
                        data["packet_type"],
                        data["previous_packet_type"],
                        data["packet_angle"],
                    )

                elif command == RcCommand.SET_HOME_POS:
                    self.set_home_pos()

                elif command == RcCommand.TRACKING_TOGGLE:
                    self.toggle_tracking(data)

                elif command == RcCommand.SET_TRACKING_POS:
                    self.set_tracking_position(data["x"], data["y"], data["z"])

                else:
                    print("[WARNING]: Wrong command send to control server")

            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA Control Server disconnected")
                break
