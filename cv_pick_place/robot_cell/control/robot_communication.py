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
        self.client = Client("opc.tcp://user:"+str(password)+"@10.35.91.101:4840/")
        self.client.connect()
        print('[INFO]: Client connected.')

    def get_nodes(self):
        """
        Using the client.get_node method, it gets nodes from OPCUA Server on PLC.
        """

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
            'ns=3;s="HMIKuka"."robot"."powerRobot"."status".override"."actualOverride"')

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

        self.PrePick_Done =  self.client.get_node(
            'ns=3;s="InstPickPlace"."instPrePickPos"."Done"')
        self.Place_Done =  self.client.get_node(
            'ns=3;s="InstPickPlace"."instPlacePos"."Done"')
            
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
            self.Robot_speed_override
        ]

        # Get values from defined nodes
        # Values are ordered in the same way as the nodes
        val = self.client.get_values(nodes)

        # Assign values from returned list to variables
        position = (round(val[0], 2),
                    round(val[1], 2),
                    round(val[2], 2),
                    round(val[3], 2),
                    round(val[4], 2),
                    round(val[5], 2),
                    val[6],
                    val[7])
        encoder_vel = round(val[8], 2)
        encoder_pos = round(val[9], 2)
        start = val[10]
        abort = val[11]
        rob_stopped = val[12]
        stop_active = val[13]
        prog_done = val[14]
        speed_override = val[15]

        return position, encoder_vel, encoder_pos, start, abort, rob_stopped, stop_active, prog_done, speed_override 

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
                position, encoder_vel, encoder_pos, start, abort, rob_stopped, stop_active, prog_done, speed_override  = self.get_robot_info()
                info_dict['pos'] = position
                info_dict['encoder_vel'] = encoder_vel
                info_dict['encoder_pos'] = encoder_pos
                info_dict['start'] = start
                info_dict['abort'] = abort
                info_dict['rob_stopped'] = rob_stopped
                info_dict['stop_active'] = stop_active
                info_dict['prog_done'] = prog_done
                info_dict['speed_override'] = speed_override
            except Exception as e:
                print('[ERROR]', e)
                print('[INFO] OPCUA disconnected')
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
                print('[ERROR]', e)
                print('[INFO] OPCUA disconnected')
                break
