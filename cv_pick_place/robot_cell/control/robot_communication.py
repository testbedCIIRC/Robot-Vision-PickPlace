import time
import multiprocessing
import multiprocessing.connection
import multiprocessing.managers

from opcua import Client


class RobotCommunication:
    """
    Class for OPCUA communication with the PLC.
    """

    def __init__(self):
        """
        RobotCommunication object constructor.
        """

    def connect_OPCUA_server(self):
        """
        Connects OPCUA Client to Server on PLC.

        """
        password = "CIIRC"
        self.client = Client("opc.tcp://user:" + str(password) + "@10.35.91.101:4840/")
        self.client.connect()
        print("[INFO]: Client connected.")

    def get_nodes(self):
        """
        Using the client.get_node() method, get all requied nodes from the PLC server.
        The node is then passed to the client.register_nodes(), which notifies the server
        that it should perform optimizations for reading / writing operations for this node.
        The client.register_nodes() only takes a list of nodes as input, and returns list of
        registered nodes as output, so single node is wrapped in a list and then received
        as first element of a list.
        """

        # fmt: off
        self.Start_Prog = self.client.register_nodes([self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."start"')])[0]
        self.Conti_Prog = self.client.register_nodes([self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."continue"')])[0]
        self.Stop_Prog = self.client.register_nodes([self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."interrupt"')])[0]
        self.Abort_Prog = self.client.register_nodes([self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."powerRobot"."command"."abort"')])[0]
        self.Prog_Done = self.client.register_nodes([self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."status"."done"')])[0]
        self.Stop_Active = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."instInterrupt"."BrakeActive"')])[0]
        self.Rob_Stopped = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instAutomaticExternal"."ROB_STOPPED"')])[0]
        self.Conveyor_Left = self.client.register_nodes([self.client.get_node(
            'ns=3;s="conveyor_left"')])[0]
        self.Conveyor_Right = self.client.register_nodes([self.client.get_node(
            'ns=3;s="conveyor_right"')])[0]
        self.Gripper_State = self.client.register_nodes([self.client.get_node(
            'ns=3;s="gripper_control"')])[0]
        self.Encoder_Vel = self.client.register_nodes([self.client.get_node(
            'ns=3;s="Encoder_1".ActualVelocity')])[0]
        self.Encoder_Pos = self.client.register_nodes([self.client.get_node(
            'ns=3;s="Encoder_1".ActualPosition')])[0]
        self.Laser_Enable = self.client.register_nodes([self.client.get_node(
            'ns=3;s="laser_field_enable"')])[0]
        self.Pick_Place_Select = self.client.register_nodes([self.client.get_node(
            'ns=3;s="pick_place_select"')])[0]
        self.Go_to_home = self.client.register_nodes([self.client.get_node(
            'ns=3;s="go_to_home"')])[0]
        self.Robot_speed_override = self.client.register_nodes([self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."powerRobot"."status".override"."actualOverride"')])[0]

        self.Act_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."X"')])[0]
        self.Act_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Y"')])[0]
        self.Act_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Z"')])[0]
        self.Act_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."A"')])[0]
        self.Act_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."B"')])[0]
        self.Act_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."C"')])[0]
        self.Act_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Turn"')])[0]
        self.Act_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstKukaControl"."instReadActualPos"."Status"')])[0]

        self.Home_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."X"')])[0]
        self.Home_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."Y"')])[0]
        self.Home_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."Z"')])[0]
        self.Home_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."A"')])[0]
        self.Home_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."B"')])[0]
        self.Home_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."C"')])[0]
        self.Home_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."Status"')])[0]
        self.Home_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[0]."E6POS"."Turn"')])[0]

        self.PrePick_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."X"')])[0]
        self.PrePick_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Y"')])[0]
        self.PrePick_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Z"')])[0]
        self.PrePick_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."A"')])[0]
        self.PrePick_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."B"')])[0]
        self.PrePick_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."C"')])[0]
        self.PrePick_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Status"')])[0]
        self.PrePick_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Turn"')])[0]

        self.Pick_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."X"')])[0]
        self.Pick_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Y"')])[0]
        self.Pick_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Z"')])[0]
        self.Pick_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."A"')])[0]
        self.Pick_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."B"')])[0]
        self.Pick_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."C"')])[0]
        self.Pick_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Status"')])[0]
        self.Pick_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Turn"')])[0]

        self.Place_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."X"')])[0]
        self.Place_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Y"')])[0]
        self.Place_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Z"')])[0]
        self.Place_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."A"')])[0]
        self.Place_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."B"')])[0]
        self.Place_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."C"')])[0]
        self.Place_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Status"')])[0]
        self.Place_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Turn"')])[0]

        self.ShPrePick_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."X"')])[0]
        self.ShPrePick_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."Y"')])[0]
        self.ShPrePick_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."Z"')])[0]
        self.ShPrePick_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."A"')])[0]
        self.ShPrePick_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."B"')])[0]
        self.ShPrePick_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."C"')])[0]
        self.ShPrePick_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."Status"')])[0]
        self.ShPrePick_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[4]."E6POS"."Turn"')])[0]

        self.ShPick_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."X"')])[0]
        self.ShPick_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."Y"')])[0]
        self.ShPick_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."Z"')])[0]
        self.ShPick_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."A"')])[0]
        self.ShPick_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."B"')])[0]
        self.ShPick_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."C"')])[0]
        self.ShPick_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."Status"')])[0]
        self.ShPick_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[5]."E6POS"."Turn"')])[0]

        self.ShPostPick_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."X"')])[0]
        self.ShPostPick_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Y"')])[0]
        self.ShPostPick_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Z"')])[0]
        self.ShPostPick_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."A"')])[0]
        self.ShPostPick_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."B"')])[0]
        self.ShPostPick_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."C"')])[0]
        self.ShPostPick_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Status"')])[0]
        self.ShPostPick_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Turn"')])[0]

        self.ShPlace_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."X"')])[0]
        self.ShPlace_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."Y"')])[0]
        self.ShPlace_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."Z"')])[0]
        self.ShPlace_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."A"')])[0]
        self.ShPlace_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."B"')])[0]
        self.ShPlace_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."C"')])[0]
        self.ShPlace_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."Status"')])[0]
        self.ShPlace_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."Turn"')])[0]

        self.ShHome_Pos_X = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."X"')])[0]
        self.ShHome_Pos_Y = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."Y"')])[0]
        self.ShHome_Pos_Z = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."Z"')])[0]
        self.ShHome_Pos_A = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."A"')])[0]
        self.ShHome_Pos_B = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."B"')])[0]
        self.ShHome_Pos_C = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."C"')])[0]
        self.ShHome_Pos_Status = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."Status"')])[0]
        self.ShHome_Pos_Turn = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."Turn"')])[0]

        self.PrePick_Done = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."instPrePickPos"."Done"')])[0]
        self.Place_Done = self.client.register_nodes([self.client.get_node(
            'ns=3;s="InstPickPlace"."instPlacePos"."Done"')])[0]
        # fmt: on

    def get_robot_info(self) -> tuple:
        """
        Reads periodically needed values from the PLC.
        To add new nodes, append requied node to the end of 'nodes' list,
        'val' list will then contain new value at the end corresponding to the values of the new node.
        Acess new values with val[15] and so on.

        Returns:
            tuple: Tuple of detected variables
        """

        # Define list of nodes
        nodes = [
            self.Act_Pos_X,
            self.Act_Pos_Y,
            self.Act_Pos_Z,
            self.Act_Pos_A,
            self.Act_Pos_B,
            self.Act_Pos_C,
            self.Act_Pos_Status,
            self.Act_Pos_Turn,
            self.Encoder_Vel,
            self.Encoder_Pos,
            self.Start_Prog,
            self.Abort_Prog,
            self.Rob_Stopped,
            self.Stop_Active,
            self.Prog_Done,
            self.Robot_speed_override,
        ]

        # Get values from defined nodes
        # Values are ordered in the same way as the nodes
        val = self.client.get_values(nodes)

        # Assign values from returned list to variables
        position = (
            round(val[0], 2),
            round(val[1], 2),
            round(val[2], 2),
            round(val[3], 2),
            round(val[4], 2),
            round(val[5], 2),
            val[6],
            val[7],
        )
        encoder_vel = round(val[8], 2)
        encoder_pos = round(val[9], 2)
        start = val[10]
        abort = val[11]
        rob_stopped = val[12]
        stop_active = val[13]
        prog_done = val[14]
        speed_override = val[15]

        return (
            position,
            encoder_vel,
            encoder_pos,
            start,
            abort,
            rob_stopped,
            stop_active,
            prog_done,
            speed_override,
        )

    def robot_server(self, info_dict: multiprocessing.managers.DictProxy):
        """
        Process to get values from PLC server.
        Periodically reads robot info from PLC and writes it into 'info_dict',
        which is dictionary read at the same time in the main process.

        Args:
            info_dict (multiprocessing.managers.DictProxy): Dictionary which is used to pass data between threads.
        """

        # Connect server and get nodes
        self.connect_OPCUA_server()
        self.get_nodes()
        time.sleep(0.5)
        while True:
            try:
                (
                    position,
                    encoder_vel,
                    encoder_pos,
                    start,
                    abort,
                    rob_stopped,
                    stop_active,
                    prog_done,
                    speed_override,
                ) = self.get_robot_info()
                info_dict["pos"] = position
                info_dict["encoder_vel"] = encoder_vel
                info_dict["encoder_pos"] = encoder_pos
                info_dict["start"] = start
                info_dict["abort"] = abort
                info_dict["rob_stopped"] = rob_stopped
                info_dict["stop_active"] = stop_active
                info_dict["prog_done"] = prog_done
                info_dict["speed_override"] = speed_override
            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA disconnected")
                break

    def encoder_server(self, encoder_pos: multiprocessing.managers.ValueProxy):
        """
        Process to get encoder value from PLC server.
        Periodically reads encoder value from PLC and writes it into 'encoder_pos.value',
        which is variable read at the same time in the main process.

        Args:
            encoder_pos (multiprocessing.managers.ValueProxy): Value which is used to pass data between threads.
        """

        # Connect server and get nodes
        self.connect_OPCUA_server()
        self.get_nodes()
        time.sleep(0.5)
        while True:
            try:
                encoder_pos.value = round(self.Encoder_Pos.get_value(), 2)
            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA disconnected")
                break
