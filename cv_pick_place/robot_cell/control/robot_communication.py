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
        self.Prog_Busy = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."example"."pickPlace"."status"."busy"')
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
        self.Robot_speed_override = self.client.get_node(
            'ns=3;s="HMIKuka"."robot"."powerRobot"."status"."override"."actualOverride"')


        self.Home_X = self.client.get_node(
            'ns=3;s="Robot_Positions"."Home"."X"')
        self.Home_Y = self.client.get_node(
            'ns=3;s="Robot_Positions"."Home"."Y"')
        self.Home_Z = self.client.get_node(
            'ns=3;s="Robot_Positions"."Home"."Z"')
        self.Home_A = self.client.get_node(
            'ns=3;s="Robot_Positions"."Home"."A"')
        self.Home_B = self.client.get_node(
            'ns=3;s="Robot_Positions"."Home"."B"')
        self.Home_C = self.client.get_node(
            'ns=3;s="Robot_Positions"."Home"."C"')
        self.Home_Status = self.client.get_node(
            'ns=3;s="Robot_Positions"."Home"."Status"')
        self.Home_Turn = self.client.get_node(
            'ns=3;s="Robot_Positions"."Home"."Turn"')

        self.PrePick_X = self.client.get_node(
            'ns=3;s="Robot_Positions"."PrePick"."X"')
        self.PrePick_Y = self.client.get_node(
            'ns=3;s="Robot_Positions"."PrePick"."Y"')
        self.PrePick_Z = self.client.get_node(
            'ns=3;s="Robot_Positions"."PrePick"."Z"')
        self.PrePick_A = self.client.get_node(
            'ns=3;s="Robot_Positions"."PrePick"."A"')
        self.PrePick_B = self.client.get_node(
            'ns=3;s="Robot_Positions"."PrePick"."B"')
        self.PrePick_C = self.client.get_node(
            'ns=3;s="Robot_Positions"."PrePick"."C"')
        self.PrePick_Status = self.client.get_node(
            'ns=3;s="Robot_Positions"."PrePick"."Status"')
        self.PrePick_Turn = self.client.get_node(
            'ns=3;s="Robot_Positions"."PrePick"."Turn"')

        self.Pick_X = self.client.get_node(
            'ns=3;s="Robot_Positions"."Pick"."X"')
        self.Pick_Y = self.client.get_node(
            'ns=3;s="Robot_Positions"."Pick"."Y"')
        self.Pick_Z = self.client.get_node(
            'ns=3;s="Robot_Positions"."Pick"."Z"')
        self.Pick_A = self.client.get_node(
            'ns=3;s="Robot_Positions"."Pick"."A"')
        self.Pick_B = self.client.get_node(
            'ns=3;s="Robot_Positions"."Pick"."B"')
        self.Pick_C = self.client.get_node(
            'ns=3;s="Robot_Positions"."Pick"."C"')
        self.Pick_Status = self.client.get_node(
            'ns=3;s="Robot_Positions"."Pick"."Status"')
        self.Pick_Turn = self.client.get_node(
            'ns=3;s="Robot_Positions"."Pick"."Turn"')

        self.PostPick_X = self.client.get_node(
            'ns=3;s="Robot_Positions"."PostPick"."X"')
        self.PostPick_Y = self.client.get_node(
            'ns=3;s="Robot_Positions"."PostPick"."Y"')
        self.PostPick_Z = self.client.get_node(
            'ns=3;s="Robot_Positions"."PostPick"."Z"')
        self.PostPick_A = self.client.get_node(
            'ns=3;s="Robot_Positions"."PostPick"."A"')
        self.PostPick_B = self.client.get_node(
            'ns=3;s="Robot_Positions"."PostPick"."B"')
        self.PostPick_C = self.client.get_node(
            'ns=3;s="Robot_Positions"."PostPick"."C"')
        self.PostPick_Status = self.client.get_node(
            'ns=3;s="Robot_Positions"."PostPick"."Status"')
        self.PostPick_Turn = self.client.get_node(
            'ns=3;s="Robot_Positions"."PostPick"."Turn"')

        self.Place_X = self.client.get_node(
            'ns=3;s="Robot_Positions"."Place"."X"')
        self.Place_Y = self.client.get_node(
            'ns=3;s="Robot_Positions"."Place"."Y"')
        self.Place_Z = self.client.get_node(
            'ns=3;s="Robot_Positions"."Place"."Z"')
        self.Place_A = self.client.get_node(
            'ns=3;s="Robot_Positions"."Place"."A"')
        self.Place_B = self.client.get_node(
            'ns=3;s="Robot_Positions"."Place"."B"')
        self.Place_C = self.client.get_node(
            'ns=3;s="Robot_Positions"."Place"."C"')
        self.Place_Status = self.client.get_node(
            'ns=3;s="Robot_Positions"."Place"."Status"')
        self.Place_Turn = self.client.get_node(
            'ns=3;s="Robot_Positions"."Place"."Turn"')
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
            self.Encoder_Vel,
            self.Encoder_Pos,
            self.Start_Prog,
            self.Abort_Prog,
            self.Rob_Stopped,
            self.Stop_Active,
            self.Prog_Busy,
            self.Robot_speed_override,
        ]
        while True:
            try:
                # Get values from defined nodes
                # Values are ordered in the same way as the nodes
                val = self.client.get_values(nodes)

                # Assign values from returned list to variables
                manag_info_dict["encoder_vel"] = round(val[0], 2)
                manag_encoder_val.value = round(val[1], 2)
                manag_info_dict["start"] = val[2]
                manag_info_dict["abort"] = val[3]
                manag_info_dict["rob_stopped"] = val[4]
                manag_info_dict["stop_active"] = val[5]
                manag_info_dict["prog_busy"] = val[6]
                manag_info_dict["speed_override"] = val[7]

            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA disconnected")
                break
