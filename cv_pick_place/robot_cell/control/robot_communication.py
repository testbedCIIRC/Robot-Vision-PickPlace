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
        address = "10.35.91.101:4840"
        timeout = 4  # Every request expects answer in this time (in seconds)
        secure_channel_timeout = 300000  # Timeout for the secure channel (in milliseconds), it should be equal to the timeout set on the PLC
        session_timeout = 30000  # Timeout for the session (in milliseconds), it should be equal to the timeout set on the PLC
        
        self.client = opcua.Client(
            "opc.tcp://user:" + str(password) + "@" + str(address) + "/",
            timeout,
        )
        self.client.secure_channel_timeout = secure_channel_timeout
        self.client.session_timeout = session_timeout
        self.client.connect()
        self.client.load_type_definitions()
        print("[INFO] OPCUA client connected to server at", str(address))

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
        self.Safe_Operational_Stop = self.client.get_node(
            'ns=3;s="Robot_Data"."Status"."Safe_Operational_Stop"')

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

        self.Pos_Start = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Start"')
        self.Pos_PrePick = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PrePick"')
        self.Pos_Pick = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Pick"')
        self.Pos_PostPick = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."PostPick"')
        self.Pos_Place = self.client.get_node(
            'ns=3;s="Robot_Data"."Pick_Place"."Positions"."Place"')
        # fmt: on

        # Register all nodes for faster read / write access.
        # The client.register_nodes() only takes a list of nodes as input, and returns list of
        # registered nodes as output, so single node is wrapped in a list and then received
        # as first element of a list.
        for _, value in self.__dict__.items():
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
            self.Conveyor_Left,
            self.Conveyor_Right,
            self.Gripper_State,
            self.Start_Prog,
            self.Conti_Prog,
            self.Prog_Busy,
            self.Prog_Interrupted,
            self.Prog_Done,
            self.Safe_Operational_Stop,
        ]

        while True:
            try:
                # Get values from defined nodes
                # Values are ordered in the same way as the nodes
                val = self.client.get_values(nodes)

                # Assign values from returned list to variables
                manag_encoder_val.value = round(val[0], 2)
                manag_info_dict["encoder_vel"] = round(val[1], 2)
                manag_info_dict["conveyor_left"] = val[2]
                manag_info_dict["conveyor_right"] = val[3]
                manag_info_dict["gripper_state"] = val[4]
                manag_info_dict["start_prog"] = val[5]
                manag_info_dict["conti_prog"] = val[6]
                manag_info_dict["prog_busy"] = val[7]
                manag_info_dict["prog_interrupted"] = val[8]
                manag_info_dict["prog_done"] = val[9]
                manag_info_dict["safe_operational_stop"] = val[10]

            except Exception as e:
                print("[ERROR]", e)
                print("[INFO] OPCUA Data Server disconnected")
                break
