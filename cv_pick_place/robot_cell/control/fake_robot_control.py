from robot_cell.control.robot_communication import RobotCommunication
from robot_cell.control.robot_control import RobotControl


class FakeNode:
    def __init__(self):
        pass

    def get_value(self):
        return 0

    def set_value(self, value):
        pass


class FakeClient:
    def __init__(self):
        pass

    def connect(self):
        pass

    def get_node(self, name):
        node = FakeNode()
        return node

    def get_values(self, nodes):
        val = []
        for node in nodes:
            val.append(0)
        return val

    def disconnect(self):
        pass


class FakeRobotCommunication(RobotCommunication):
    def connect_OPCUA_server(self):
        """
        Connects OPC UA Client to Server on PLC.

        """
        self.client = FakeClient()
        print('[INFO]: Client connected.')


class FakeRobotControl(RobotControl):
    def connect_OPCUA_server(self):
        """
        Connects OPC UA Client to Server on PLC.

        """
        self.client = FakeClient()
        print('[INFO]: Client connected.')
