import time
import multiprocessing
import multiprocessing.connection
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
    PICK_PLACE_SELECT = 13


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
        time.sleep(0.5)
        self.Conti_Prog.set_value(ua.DataValue(False))
        if self.verbose:
            print("[INFO]: Program continued.")

    def stop_program(self):
        """
        Stop robot action.
        """

        self.Stop_Prog.set_value(ua.DataValue(True))
        print("[INFO]: Program interrupted.")
        time.sleep(0.5)
        self.Stop_Prog.set_value(ua.DataValue(False))

    def abort_program(self):
        """
        Abort robot action.
        """

        self.Abort_Prog.set_value(ua.DataValue(True))
        print("[INFO]: Program aborted.")
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
        if self.verbose:
            print("[INFO]: Program started.")
        time.sleep(0.5)
        self.Start_Prog.set_value(ua.DataValue(False))
        time.sleep(0.5)

    def close_program(self):
        """
        Close robot program.
        """

        self.Abort_Prog.set_value(ua.DataValue(True))
        print("[INFO]: Program aborted.")
        self.Abort_Prog.set_value(ua.DataValue(False))
        self.Conti_Prog.set_value(ua.DataValue(False))
        self.client.disconnect()
        print("[INFO]: Client disconnected.")
        time.sleep(0.5)

    def change_gripper_state(self, state: bool):
        """
        Switch gripper on/off.

        Args:
            state (bool): If the gripper should be turned on or off.
        """

        self.Gripper_State.set_value(ua.DataValue(state))
        if self.verbose:
            print("[INFO]: Gripper state is {}.".format(state))
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
        if self.verbose:
            print("[INFO]: Sent robot to home pos.")
        time.sleep(0.4)
        self.Go_to_home.set_value(ua.DataValue(False))
        time.sleep(0.4)

    def change_trajectory(
        self,
        x: float,
        y: float,
        rot: float,
        packet_type: int,
        x_offset: float = 0.0,
        pack_z: int = 5,
    ):
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

        # fmt: off
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
        # fmt: on

        self.client.set_values(nodes, values)

        time.sleep(0.7)

    def change_trajectory_short(
        self,
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
        c: float = 180.0,
    ):
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

        # fmt: off
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
        # fmt: on

        self.client.set_values(nodes, values)

        time.sleep(0.2)

    def set_home_pos_short(self):
        """
        Set home position for short pick place to position in dictionary.
        """

        nodes = []
        values = []

        # fmt: off
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
        # fmt: on

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
                    self.change_trajectory(
                        data["x"],
                        data["y"],
                        data["rot"],
                        data["packet_type"],
                        x_offset=data["x_offset"],
                        pack_z=data["pack_z"],
                    )

                elif command == RcCommand.CHANGE_SHORT_TRAJECTORY:
                    self.change_trajectory_short(
                        data["x"],
                        data["y"],
                        data["rot"],
                        data["packet_type"],
                        x_offset=data["x_offset"],
                        pack_z=data["pack_z"],
                        a=data["a"],
                        b=data["b"],
                        c=data["c"],
                        z_offset=data["z_offset"],
                    )

                elif command == RcCommand.SET_HOME_POS_SH:
                    self.set_home_pos_short()

                elif command == RcCommand.GO_TO_HOME:
                    self.go_to_home()

                elif command == RcCommand.PICK_PLACE_SELECT:
                    self.Pick_Place_Select.set_value(ua.DataValue(data))

                else:
                    print("[WARNING]: Wrong command send to control server")

            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA disconnected")
                break
