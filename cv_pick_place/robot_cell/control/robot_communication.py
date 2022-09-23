import time
import multiprocessing
import multiprocessing.connection
import multiprocessing.managers

import opcua


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
        self.client = opcua.Client(
            "opc.tcp://user:" + str(password) + "@10.35.91.101:4840/"
        )
        self.client.connect()
        print("[INFO]: Client connected.")

    def get_nodes(self):
        """
        Using the client.get_node() method, get all requied nodes from the PLC server.
        All nodes are then passed to the client.register_nodes() method, which notifies the server
        that it should perform optimizations for reading / writing operations for these nodes.
        """

        # fmt: off
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
        self.Go_to_home = self.client.get_node(
            'ns=3;s="go_to_home"')
        self.Robot_speed_override = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."powerRobot"."status"."override"."actualOverride"')

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

        self.ShPostPick_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."X"')
        self.ShPostPick_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Y"')
        self.ShPostPick_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Z"')
        self.ShPostPick_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."A"')
        self.ShPostPick_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."B"')
        self.ShPostPick_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."C"')
        self.ShPostPick_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Status"')
        self.ShPostPick_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[6]."E6POS"."Turn"')

        self.ShPlace_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."X"')
        self.ShPlace_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."Y"')
        self.ShPlace_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."Z"')
        self.ShPlace_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."A"')
        self.ShPlace_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."B"')
        self.ShPlace_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."C"')
        self.ShPlace_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."Status"')
        self.ShPlace_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[7]."E6POS"."Turn"')

        self.ShHome_Pos_X = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."X"')
        self.ShHome_Pos_Y = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."Y"')
        self.ShHome_Pos_Z = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."Z"')
        self.ShHome_Pos_A = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."A"')
        self.ShHome_Pos_B = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."B"')
        self.ShHome_Pos_C = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."C"')
        self.ShHome_Pos_Status = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."Status"')
        self.ShHome_Pos_Turn = self.client.get_node(
            'ns=3;s="InstPickPlace"."positions"[8]."E6POS"."Turn"')

        self.PrePick_Done = self.client.get_node(
            'ns=3;s="InstPickPlace"."instPrePickPos"."Done"')
        self.Place_Done = self.client.get_node(
            'ns=3;s="InstPickPlace"."instPlacePos"."Done"')
        # fmt: on

        # Register all nodes for faster read / write access.
        # The client.register_nodes() only takes a list of nodes as input, and returns list of
        # registered nodes as output, so single node is wrapped in a list and then received
        # as first element of a list.
        for key, value in self.__dict__.items():
            if type(value) == opcua.Node:
                value = self.client.register_nodes([value])[0]

    def robot_server(
        self,
        manag_info_dict: multiprocessing.managers.DictProxy,
        manag_encoder_val: multiprocessing.managers.ValueProxy,
    ):
        """
        Process to get values from PLC server.
        Periodically reads robot info from PLC and writes it into 'manag_info_dict', and 'manag_encoder_val.value'
        which is dictionary read at the same time in the main process.

        Args:
            manag_info_dict (multiprocessing.managers.DictProxy): Dictionary which is used to pass data between threads.
            manag_encoder_val (multiprocessing.managers.ValueProxy): Value object from multiprocessing Manager for passing encoder value to another process.
        """

        # Connect server and get nodes
        self.connect_OPCUA_server()
        self.get_nodes()
        time.sleep(0.5)

        # Define list of nodes read by this server
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
        while True:
            try:
                # Get values from defined nodes
                # Values are ordered in the same way as the nodes
                val = self.client.get_values(nodes)

                # Assign values from returned list to variables
                manag_info_dict["pos"] = (
                    round(val[0], 2),  # X
                    round(val[1], 2),  # Y
                    round(val[2], 2),  # Z
                    round(val[3], 2),  # A
                    round(val[4], 2),  # B
                    round(val[5], 2),  # X
                    val[6],  # Robot Status
                    val[7],  # Robot Turn
                )
                manag_info_dict["encoder_vel"] = round(val[8], 2)
                manag_encoder_val.value = round(val[9], 2)
                manag_info_dict["start"] = val[10]
                manag_info_dict["abort"] = val[11]
                manag_info_dict["rob_stopped"] = val[12]
                manag_info_dict["stop_active"] = val[13]
                manag_info_dict["prog_done"] = val[14]
                manag_info_dict["speed_override"] = val[15]

            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA disconnected")
                break
