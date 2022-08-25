import os
import json

from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Pipe

from robot_cell.control.pick_place_demos import RobotDemos
from mult_packets_pick_place import main_multi_packets
from robot_cell.control.robot_control import RobotControl
from robot_cell.control.robot_communication import RobotCommunication

ROB_CONFIG_FILE = os.path.join("config", "robot_config.json")


def program_mode(
    demos: RobotDemos,
    r_control: RobotControl,
    r_comm_info: RobotCommunication,
    r_comm_encoder: RobotCommunication,
    rob_config: dict,
) -> None:
    """
    Program selection function.
    Allows user to select a demo and starts appropriate functions.
    At the end this function starts again, recursively.

    Args:
        demos (RobotDemos): RobotDemos object containing pick and place demo functions.
        r_control (RobotControl): RobotControl for controlling the robot cell with commands.
        r_comm_info (RobotCommunication): RobotCommunication object for reading robot cell information.
        r_comm_encoder (RobotCommunication): RobotCommunication object for reading encoder values.
        rob_config (dict): Dictionary with parameters setting the behaviour of the cell.
    """

    # Show message about robot programs
    print(
        "Select pick and place mode: \n"
        + "1 : Pick and place with static conveyor and hand gestures\n"
        + "2 : Pick and place with static conveyor and multithreading\n"
        + "3 : Pick and place with moving conveyor, point cloud and multithreading\n"
        + "4 : Main Pick and place\n"
        + "e : To exit program\n"
    )

    # Read mode input
    mode = input()

    # Dictionary with robot positions and robot programs
    # fmt: off
    modes_dict = {
        "1": {
            "dict": Pick_place_dict,
            "func": demos.main_robot_control_demo
        },
        "2": {
            "dict": Pick_place_dict,
            "func": demos.main_pick_place
        },
        "3": {
            "dict": Pick_place_dict_conv_mov,
            "func": demos.main_pick_place_conveyor_w_point_cloud,
        },
        "4": {
            "dict": Short_pick_place_dict,
            "func": main_multi_packets
        },
    }
    # fmt: on

    # If mode is a program key
    if mode in modes_dict:
        # Set robot positions and robot program
        r_control.rob_dict = modes_dict[mode]["dict"]
        robot_prog = modes_dict[mode]["func"]

        # If first mode (not threaded) was selected
        if mode == "1":
            print(
                "[INFO]: Starting program: Pick and place with static conveyor and hand gestures"
            )
            robot_prog(r_control)

        # If fourth mode (has extra processes) was selected
        elif mode == "4":
            with Manager() as manager:
                print("[INFO]: Starting program: Multi packets pick and place")

                # Create Pipe and Manager objects for passing data between processes
                info_dict = manager.dict()
                encoder_pos = manager.Value("d", None)
                control_pipe_1, control_pipe_2 = Pipe()

                # Create and start processes
                main_proc = Process(
                    target=robot_prog,
                    args=(
                        rob_config,
                        r_control.rob_dict,
                        info_dict,
                        encoder_pos,
                        control_pipe_1,
                    ),
                )
                info_server_proc = Process(
                    target=r_comm_info.robot_server,
                    args=(info_dict,),
                )
                encoder_server_proc = Process(
                    target=r_comm_encoder.encoder_server,
                    args=(encoder_pos,),
                )
                control_server_proc = Process(
                    target=r_control.control_server,
                    args=(control_pipe_2,),
                )

                main_proc.start()
                info_server_proc.start()
                encoder_server_proc.start()
                control_server_proc.start()

                # Wait for the main process to end
                main_proc.join()
                info_server_proc.kill()
                encoder_server_proc.kill()
                control_server_proc.kill()

        # Otherwise start selected threaded program
        else:
            with Manager() as manager:
                print(f"[INFO]: Starting program: {mode}")

                # Create Pipe and Manager objects for passing data between processes
                info_dict = manager.dict()

                # Create and start processes
                main_proc = Process(
                    target=robot_prog,
                    args=(
                        r_control,
                        rob_config["MODEL"]["PATHS"],
                        rob_config["MODEL"]["FILES"],
                        rob_config["MODEL"]["CHECK_POINT"],
                        info_dict,
                    ),
                )
                info_server_proc = Process(
                    target=r_comm_info.robot_server,
                    args=(info_dict,),
                )

                main_proc.start()
                info_server_proc.start()

                # Wait for the main process to end
                main_proc.join()
                info_server_proc.kill()

    # If input is exit, exit python
    if mode == "e":
        exit()

    # Return function recursively
    return program_mode(demos, r_control, r_comm_info, r_comm_encoder, rob_config)


if __name__ == "__main__":
    # Read robot configuration
    with open(ROB_CONFIG_FILE) as file:
        rob_config = json.load(file)

    # Define robot positions dictionaries from json file
    with open(rob_config["PATHS"]["ROBOT_POSITIONS_FILE"]) as file:
        robot_poses = json.load(file)

    Pick_place_dict_conv_mov_slow = robot_poses["Pick_place_dict_conv_mov_slow"]
    Pick_place_dict_conv_mov = robot_poses["Pick_place_dict_conv_mov"]
    Pick_place_dict = robot_poses["Pick_place_dict"]
    Short_pick_place_dict = robot_poses["Short_pick_place_dict"]

    # Initialize robot demos and robot control objects
    r_control = RobotControl(None)
    r_comm_info = RobotCommunication()
    r_comm_encoder = RobotCommunication()
    demos = RobotDemos(
        rob_config["MODEL"]["PATHS"],
        rob_config["MODEL"]["FILES"],
        rob_config["MODEL"]["CHECK_POINT"],
    )

    # Start program mode selection
    program_mode(demos, r_control, r_comm_info, r_comm_encoder, rob_config)
