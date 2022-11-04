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
            'ns=3;s="Robot_Data"."Pick_Place"."Control"."Start"')
        self.Conti_Prog = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Control"."Continue"')
        self.Prog_Done = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Status"."Done"')
        self.Prog_Busy = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Status"."Busy"')
        self.Prog_Interrupted = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Status"."Interrupted"')

        self.Conveyor_Left = self.client.get_node(
            'ns=3;s="Program_Data"."Conveyor_Left_Toggle"')
        self.Conveyor_Right = self.client.get_node(
            'ns=3;s="Program_Data"."Conveyor_Right_Toggle"')
        self.Gripper_State = self.client.get_node(
            'ns=3;s="Program_Data"."Gripper_Toggle"')
        self.Encoder_Vel = self.client.get_node(
            'ns=3;s="Encoder_1".ActualVelocity')
        self.Encoder_Pos = self.client.get_node(
            'ns=3;s="Encoder_1".ActualPosition')

        self.Start_X = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."X"')
        self.Start_Y = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."Y"')
        self.Start_Z = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."Z"')
        self.Start_A = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."A"')
        self.Start_B = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."B"')
        self.Start_C = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."C"')
        self.Start_Status = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."Status"')
        self.Start_Turn = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"."Turn"')

        self.PrePick_X = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"."X"')
        self.PrePick_Y = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"."Y"')
        self.PrePick_Z = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"."Z"')
        self.PrePick_A = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"."A"')
        self.PrePick_B = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"."B"')
        self.PrePick_C = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"."C"')
        self.PrePick_Status = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"."Status"')
        self.PrePick_Turn = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"."Turn"')

        self.Pick_X = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"."X"')
        self.Pick_Y = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"."Y"')
        self.Pick_Z = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"."Z"')
        self.Pick_A = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"."A"')
        self.Pick_B = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"."B"')
        self.Pick_C = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"."C"')
        self.Pick_Status = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"."Status"')
        self.Pick_Turn = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"."Turn"')

        self.PostPick_X = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"."X"')
        self.PostPick_Y = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"."Y"')
        self.PostPick_Z = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"."Z"')
        self.PostPick_A = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"."A"')
        self.PostPick_B = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"."B"')
        self.PostPick_C = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"."C"')
        self.PostPick_Status = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"."Status"')
        self.PostPick_Turn = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"."Turn"')

        self.Place_X = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"."X"')
        self.Place_Y = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"."Y"')
        self.Place_Z = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"."Z"')
        self.Place_A = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"."A"')
        self.Place_B = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"."B"')
        self.Place_C = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"."C"')
        self.Place_Status = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"."Status"')
        self.Place_Turn = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"."Turn"')
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
            self.Encoder_Pos,
            self.Encoder_Vel,
            self.Prog_Busy,
            self.Prog_Interrupted,
            self.Prog_Done,
        ]

        while True:
            try:
                # Get values from defined nodes
                # Values are ordered in the same way as the nodes
                val = self.client.get_values(nodes)

                # Assign values from returned list to variables
                manag_encoder_val.value = round(val[0], 2)
                manag_info_dict["encoder_vel"] = round(val[1], 2)
                manag_info_dict["prog_busy"] = val[2]
                manag_info_dict["prog_interrupted"] = val[3]
                manag_info_dict["prog_done"] = val[4]

            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA Data Server disconnected")
                break
