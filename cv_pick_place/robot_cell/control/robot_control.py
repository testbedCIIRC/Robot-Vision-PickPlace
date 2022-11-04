import time
import multiprocessing
import multiprocessing.connection
from enum import Enum, unique, auto

import cv2
import numpy as np
from opcua import ua

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

        self.Conti_Prog.set_value(ua.DataValue(True))
        if self.verbose:
            print("[INFO]: Program continued.")

    def start_program(self, mode: bool = False):
        """
        Start robot program.

        Args:
            mode (bool): PLC program mode selection.
        """

        self.Start_Prog.set_value(ua.DataValue(True))
        if self.verbose:
            print("[INFO]: Program started.")

    def close_program(self):
        """
        Close robot program.
        """

        self.client.disconnect()
        print("[INFO]: Client disconnected.")

    def change_gripper_state(self, state: bool):
        """
        Switch gripper on/off.

        Args:
            state (bool): If the gripper should be turned on or off.
        """

        self.Gripper_State.set_value(ua.DataValue(state))
        if self.verbose:
            print("[INFO]: Gripper state is {}.".format(state))

    def change_conveyor_right(self, conv_right: bool):
        """
        Switch conveyor right direction on/off.

        Args:
            conv_right (bool): If the conveyor should be turned on or off in the right direction.
        """

        self.Conveyor_Right.set_value(ua.DataValue(conv_right))

    def change_conveyor_left(self, conv_left: bool):
        """
        Switch conveyor left direction on/off.

        Args:
            conv_left (bool): If the conveyor should be turned on or off in the left direction.
        """

        self.Conveyor_Left.set_value(ua.DataValue(conv_left))

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
    ):
        """
        Updates the trajectory points for the robot program.

        Args:
            x (float): The pick x coordinate of the packet.
            x_offset (float): X offset between prepick and pick position.
            y (float): The pick y coordinate of the packet.
            z (float): Pick height.
            z_offset (float): Z height offset from pick height for all positions except for pick position.
            a (float): Angle in degrees.
            b (float): Angle in degrees.
            c (float): Angle in degrees.
            packet_type (int): The detected packet class.
        """

        nodes = []
        values = []

        # fmt: off
        nodes.append(self.Start_X)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][previous_packet_type]['x'], ua.VariantType.Float)))
        nodes.append(self.Start_Y)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][previous_packet_type]['y'] - 50, ua.VariantType.Float)))
        nodes.append(self.Start_Z)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][previous_packet_type]['z'], ua.VariantType.Float)))
        nodes.append(self.Start_A)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][previous_packet_type]['a'], ua.VariantType.Float)))
        nodes.append(self.Start_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][previous_packet_type]['b'], ua.VariantType.Float)))
        nodes.append(self.Start_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][previous_packet_type]['c'], ua.VariantType.Float)))
        nodes.append(self.Start_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][previous_packet_type]['status'], ua.VariantType.Int16)))
        nodes.append(self.Start_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][previous_packet_type]['turn'], ua.VariantType.Int16)))

        nodes.append(self.PrePick_X)
        values.append(ua.DataValue(ua.Variant(x, ua.VariantType.Float)))
        nodes.append(self.PrePick_Y)
        values.append(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        nodes.append(self.PrePick_Z)
        values.append(ua.DataValue(ua.Variant(z + z_offset, ua.VariantType.Float)))
        nodes.append(self.PrePick_A)
        values.append(ua.DataValue(ua.Variant(a, ua.VariantType.Float)))
        nodes.append(self.PrePick_B)
        values.append(ua.DataValue(ua.Variant(b, ua.VariantType.Float)))
        nodes.append(self.PrePick_C)
        values.append(ua.DataValue(ua.Variant(c, ua.VariantType.Float)))
        nodes.append(self.PrePick_Status)
        values.append(ua.DataValue(ua.Variant(2, ua.VariantType.Int16)))
        nodes.append(self.PrePick_Turn)
        values.append(ua.DataValue(ua.Variant(0, ua.VariantType.Int16)))

        nodes.append(self.Pick_X)
        values.append(ua.DataValue(ua.Variant(x + x_offset, ua.VariantType.Float)))
        nodes.append(self.Pick_Y)
        values.append(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        nodes.append(self.Pick_Z)
        values.append(ua.DataValue(ua.Variant(z, ua.VariantType.Float)))
        nodes.append(self.Pick_A)
        values.append(ua.DataValue(ua.Variant(a, ua.VariantType.Float)))
        nodes.append(self.Pick_B)
        values.append(ua.DataValue(ua.Variant(b, ua.VariantType.Float)))
        nodes.append(self.Pick_C)
        values.append(ua.DataValue(ua.Variant(c, ua.VariantType.Float)))
        nodes.append(self.Pick_Status)
        values.append(ua.DataValue(ua.Variant(2, ua.VariantType.Int16)))
        nodes.append(self.Pick_Turn)
        values.append(ua.DataValue(ua.Variant(0, ua.VariantType.Int16)))

        nodes.append(self.PostPick_X)
        values.append(ua.DataValue(ua.Variant(x + 1.5 * x_offset, ua.VariantType.Float)))
        nodes.append(self.PostPick_Y)
        values.append(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        nodes.append(self.PostPick_Z)
        values.append(ua.DataValue(ua.Variant(z + z_offset, ua.VariantType.Float)))
        nodes.append(self.PostPick_A)
        values.append(ua.DataValue(ua.Variant(a, ua.VariantType.Float)))
        nodes.append(self.PostPick_B)
        values.append(ua.DataValue(ua.Variant(b, ua.VariantType.Float)))
        nodes.append(self.PostPick_C)
        values.append(ua.DataValue(ua.Variant(c, ua.VariantType.Float)))
        nodes.append(self.PostPick_Status)
        values.append(ua.DataValue(ua.Variant(2, ua.VariantType.Int16)))
        nodes.append(self.PostPick_Turn)
        values.append(ua.DataValue(ua.Variant(0, ua.VariantType.Int16)))

        nodes.append(self.Place_X)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['x'], ua.VariantType.Float)))
        nodes.append(self.Place_Y)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['y'], ua.VariantType.Float)))
        nodes.append(self.Place_Z)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['z'], ua.VariantType.Float)))
        nodes.append(self.Place_A)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['a'], ua.VariantType.Float)))
        nodes.append(self.Place_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['b'], ua.VariantType.Float)))
        nodes.append(self.Place_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['c'], ua.VariantType.Float)))
        nodes.append(self.Place_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['status'], ua.VariantType.Int16)))
        nodes.append(self.Place_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['place_pos'][packet_type]['turn'], ua.VariantType.Int16)))
        # fmt: on

        self.client.set_values(nodes, values)

    def set_home_pos(self):
        """
        Set home position for short pick place to position in dictionary.
        """

        nodes = []
        values = []

        # fmt: off
        nodes.append(self.Start_X)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['x'], ua.VariantType.Float)))
        nodes.append(self.Start_Y)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['y'], ua.VariantType.Float)))
        nodes.append(self.Start_Z)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['z'], ua.VariantType.Float)))
        nodes.append(self.Start_A)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['a'], ua.VariantType.Float)))
        nodes.append(self.Start_B)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['b'], ua.VariantType.Float)))
        nodes.append(self.Start_C)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['c'], ua.VariantType.Float)))
        nodes.append(self.Start_Status)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['status'], ua.VariantType.Int16)))
        nodes.append(self.Start_Turn)
        values.append(ua.DataValue(ua.Variant(self.rob_dict['home_pos'][0]['turn'], ua.VariantType.Int16)))
        # fmt: on

        self.client.set_values(nodes, values)

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
        self.get_nodes()
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

                elif command == RcCommand.CONTINUE_PROGRAM:
                    self.continue_program()

                elif command == RcCommand.CLOSE_PROGRAM:
                    self.close_program()

                elif command == RcCommand.START_PROGRAM:
                    self.start_program(data)

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
                    )

                elif command == RcCommand.SET_HOME_POS:
                    self.set_home_pos()

                else:
                    print("[WARNING]: Wrong command send to control server")

            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA Control Server disconnected")
                break
