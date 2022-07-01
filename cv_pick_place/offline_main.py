import json
from multiprocessing import Process
from multiprocessing import Pipe

from robot_cell.control.fake_robot_control import FakeRobotCommunication, FakeRobotControl
from robot_cell.control.robot_communication import RobotCommunication
from robot_cell.control.robot_control import RobotControl
from cv_pick_place_main import main

if __name__ == '__main__':
    # Define robot positions dictionaries from json file.
    file = open('robot_positions.json')
    robot_poses = json.load(file)
    Pick_place_dict_conv_mov_slow = robot_poses['Pick_place_dict_conv_mov_slow']
    Pick_place_dict_conv_mov = robot_poses['Pick_place_dict_conv_mov']
    Pick_place_dict = robot_poses['Pick_place_dict']
    
    # Initialize robot control objects
    r_ctrl = FakeRobotControl(Pick_place_dict_conv_mov)
    r_comm = FakeRobotCommunication()

    # Start threads
    connection_1, connection_2 = Pipe()
    proc_server = Process(target = r_comm.robot_server, args = (connection_1, ))
    proc_server.start()

    # Wait for threads to end
    main(r_ctrl, connection_2)
    proc_server.kill()
