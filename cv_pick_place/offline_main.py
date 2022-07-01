import json
from multiprocessing import Process
from multiprocessing import Pipe

from robot_cell.control.fake_robot_control import FakeRobotCommunication, FakeRobotControl
from robot_cell.control.robot_communication import RobotCommunication
from robot_cell.control.robot_control import RobotControl
from cv_pick_place_main import main

if __name__ == '__main__':
    # Define robot positions dictionaries from json file
    file = open('robot_positions.json')
    robot_poses = json.load(file)
    Pick_place_dict_conv_mov_slow = robot_poses['Pick_place_dict_conv_mov_slow']
    Pick_place_dict_conv_mov = robot_poses['Pick_place_dict_conv_mov']
    Pick_place_dict = robot_poses['Pick_place_dict']
    
    # Initialize robot control objects
    r_ctrl = FakeRobotControl(Pick_place_dict_conv_mov)
    r_comm = FakeRobotCommunication()

    # Create processes and connections
    connection_1, connection_2 = Pipe()
    main_proc = Process(target = main, args = (r_ctrl, connection_1))
    info_server_proc = Process(target = r_comm.robot_server, args = (connection_2, ))

    # Start processes
    main_proc.start()
    info_server_proc.start()

    # Wait for main process to end and kill the server processes
    main_proc.join()
    info_server_proc.kill()
