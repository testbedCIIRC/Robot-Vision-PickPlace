import os
import json
import argparse

from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Pipe

from mult_packets_pick_place import main_multi_packets
from tracking_program import tracking_program
from robot_cell.control.robot_control import RobotControl
from robot_cell.control.robot_communication import RobotCommunication

ROB_CONFIG_FILE = os.path.join("config", "robot_config.json")


def bool_str(string: str) -> bool:
    """
    Used in argument parser to detect boolean flags written as a string.

    Args:
        string (str): String to be evaluated.

    Returns:
        bool: True, False, depending on contents of the string.

    Raises:
        argparse.ArgumentTypeError: Error in case the string does not contain any of the expected values.
    """

    if string in ["True", "true"]:
        return True
    elif string in ["False", "false"]:
        return False
    else:
        raise argparse.ArgumentTypeError


if __name__ == "__main__":
    # Read config file path as command line argument
    parser = argparse.ArgumentParser(description="Robot cell input arguments.")
    parser.add_argument(
        "--config-file",
        default=ROB_CONFIG_FILE,
        type=str,
        dest="CONFIG_FILE",
        help="Path to configuration file",
    )
    config, _ = parser.parse_known_args()

    # Read config file using provided path
    with open(config.CONFIG_FILE, "r") as file:
        rob_config = json.load(file)

    # Read all other input arguments as specified inside the config file
    for param in rob_config.items():
        if isinstance(param[1]["default"], bool):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                type=bool_str,
            )
        elif isinstance(param[1]["default"], list):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                nargs=len(param[1]["default"]),
                type=type(param[1]["default"][0]),
            )
        elif isinstance(param[1]["default"], int):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                type=int,
            )
        elif isinstance(param[1]["default"], float):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                type=float,
            )
        elif isinstance(param[1]["default"], str):
            parser.add_argument(
                param[1]["arg"],
                default=param[1]["default"],
                dest=param[0],
                help=param[1]["help"],
                type=str,
            )
        else:
            print(f"[WARNING] Default value of {param[0]} config parameter not handled")
    rob_config = parser.parse_args()

    # Read robot positions dictionaries from json file
    with open(rob_config.path_robot_positions) as file:
        robot_poses = json.load(file)

    # Create dictionary with program modes, functions and robot positions they use
    modes_dict = {
        "1": {
            "help": "Object sorting with moving conveyor",
            "dict": robot_poses["short_pick_place_dict"],
            "func": main_multi_packets,
        },
        "2": {
            "help": "Tracking packet with robot gripper (Does not work)",
            "dict": robot_poses["short_pick_place_dict"],
            "func": tracking_program,
        },
    }

    # Initialize robot demos and robot control objects
    r_control = RobotControl(None)
    r_comm_info = RobotCommunication()

    # Start program mode selection
    while True:
        # Read mode input
        if rob_config.mode in modes_dict:
            mode = rob_config.mode
            rob_config.mode = "0"
        else:
            # Show message about robot programs
            print("Select pick and place mode:")
            for mode in modes_dict:
                print(f"{mode} : {modes_dict[mode]['help']}")
            print("e : Exit")
            mode = input()

        # If mode is a program key
        if mode in modes_dict:
            # Set robot positions and robot program
            r_control.rob_dict = modes_dict[mode]["dict"]
            robot_prog = modes_dict[mode]["func"]
            print(f"[INFO] Starting mode {mode} ({modes_dict[mode]['help']})")

            with Manager() as manager:
                # Dictionary Manager for passing data between processes
                manag_info_dict = manager.dict()

                # Value Manager separate from dictionary for passing encoder value between processes
                # Encoder value needs to be updated more often
                manag_encoder_val = manager.Value("d", None)

                # Pipe objects for passing commands from main to robot server
                # One object goes into main, the other into the server
                # Pipe is buffered, which makes it better than Manager in this case
                pipe_main, pipe_server = Pipe()

                # Create and start processes
                main_proc = Process(
                    target=robot_prog,
                    args=(
                        rob_config,
                        r_control.rob_dict,
                        manag_info_dict,
                        manag_encoder_val,
                        pipe_main,
                    ),
                )
                info_server_proc = Process(
                    target=r_comm_info.robot_server,
                    args=(manag_info_dict, manag_encoder_val),
                )
                control_server_proc = Process(
                    target=r_control.control_server,
                    args=(pipe_server,),
                )

                main_proc.start()
                info_server_proc.start()
                control_server_proc.start()

                # Wait for the main process to end
                main_proc.join()
                info_server_proc.kill()
                control_server_proc.kill()

            if rob_config.auto_exit:
                exit()

        # If input is exit, exit python
        elif mode == "e":
            exit()
