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
    r_control = FakeRobotControl(Pick_place_dict_conv_mov)
    r_comm_info = FakeRobotCommunication()
    r_comm_encoder = FakeRobotCommunication()

    # Create processes and connections
    pipe_1_in, pipe_1_out = Pipe()
    pipe_2_in, pipe_2_out = Pipe()
    main_proc = Process(target = main, args = (r_control, None, None, None, pipe_1_out, pipe_2_out))
    info_server_proc = Process(target = r_comm_info.robot_server, args = (pipe_1_in, ))
    encoder_server_proc = Process(target = r_comm_encoder.encoder_server, args = (pipe_2_in, ))

    main_proc.start()
    info_server_proc.start()
    encoder_server_proc.start()

    # Wait for the main process to end
    main_proc.join()
    info_server_proc.kill()
    encoder_server_proc.kill()
