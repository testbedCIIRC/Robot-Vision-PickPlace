import json
from multiprocessing import Process
from multiprocessing import Pipe
from multiprocessing import Manager

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
    with Manager() as manager:
            info_dict = manager.dict()
            encoder_pos = manager.Value('d', 0.0)

            control_pipe_1, control_pipe_2 = Pipe()

            main_proc = Process(target = main, args = (r_control.rob_dict['pick_pos_base'][0]['x'], None, None, None, info_dict, encoder_pos, control_pipe_1))
            info_server_proc = Process(target = r_comm_info.robot_server, args = (info_dict, ))
            encoder_server_proc = Process(target = r_comm_encoder.encoder_server, args = (encoder_pos, ))
            control_server_proc = Process(target = r_control.control_server, args = (control_pipe_2, ))

            main_proc.start()
            info_server_proc.start()
            encoder_server_proc.start()
            control_server_proc.start()

            # Wait for the main process to end
            main_proc.join()
            info_server_proc.kill()
            encoder_server_proc.kill()
            control_server_proc.kill()