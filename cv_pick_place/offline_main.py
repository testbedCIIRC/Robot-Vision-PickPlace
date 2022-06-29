import json
from queue import Queue
from threading import Thread

from robot_cell.control.fake_robot_control import FakeRobotControl
from cv_pick_place_main import main

if __name__ == '__main__':
    # Define robot positions dictionaries from json file.
    file = open('robot_positions.json')
    robot_poses = json.load(file)
    Pick_place_dict_conv_mov_slow = robot_poses['Pick_place_dict_conv_mov_slow']
    Pick_place_dict_conv_mov = robot_poses['Pick_place_dict_conv_mov']
    Pick_place_dict = robot_poses['Pick_place_dict']
    
    # Initialize robot demos and robot control objects.
    rc = FakeRobotControl(None)
    rc.rob_dict = Pick_place_dict_conv_mov

    # Start threads
    q = Queue(maxsize = 1)
    t1 = Thread(target = main, args =(rc, q))
    t2 = Thread(target = rc.robot_server, args =(q, ))
    t1.start()
    t2.start()

    # Wait for threads to end
    t1.join()
