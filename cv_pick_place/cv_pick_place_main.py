import os
import cv2 
import json
import time
import numpy as np
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Pipe

from robot_cell.packet.packet_object import Packet
from robot_cell.packet.item_object import Item
from robot_cell.packet.packettracker import PacketTracker
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.packet.point_cloud_viz import PointCloudViz
from robot_cell.packet.centroidtracker import CentroidTracker
from robot_cell.control.robot_communication import RobotCommunication
from robot_cell.control.robot_control import RobotControl
from robot_cell.control.robot_control import RcCommand
from robot_cell.control.robot_control import RcData
from robot_cell.control.pick_place_demos import RobotDemos
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.packet_detector import PacketDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag

from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.packet.item_tracker import ItemTracker
from robot_cell.functions import *


# DETECTOR_TYPE = 'deep_1'
#DETECTOR_TYPE = 'deep_2'
DETECTOR_TYPE = 'hsv'


def main(rob_dict, paths, files, check_point, info_dict, encoder_pos_m, control_pipe):
    """
    Thread for pick and place with moving conveyor and point cloud operations.
    
    Parameters:
    rc (object): RobotControl object for program execution.
    server_in (object): Queue object containing data from the PLC server.
    
    """
    # Inititalize objects
    apriltag = ProcessingApriltag()
    pt = ItemTracker(max_disappeared_frames = 20, guard = 50, max_item_distance = 400)
    dc = DepthCamera(config_path = 'D435_camera_config.json')

    if DETECTOR_TYPE == 'deep_1':
        show_boot_screen('STARTING NEURAL NET...')
        pack_detect = PacketDetector(paths, files, check_point)
    elif DETECTOR_TYPE == 'deep_2':
        # TODO Implement new deep detector
        pass
    elif DETECTOR_TYPE == 'hsv':
        pack_detect = ThresholdDetector(ignore_vertical_px = 133, ignore_horizontal_px = 50, max_ratio_error = 0.15,
                                        white_lower = [60, 0, 85], white_upper = [179, 255, 255],
                                        brown_lower = [0, 33, 57], brown_upper = [60, 255, 178])

    # Drawing toggles
    show_bbox = True # Bounding box visualization enable
    show_frame_data = False # Show frame data (robot pos, encoder vel, FPS ...)
    show_depth_map = False # Overlay colorized depth enable
    show_hsv_mask = False # Remove pixels not within HSV mask boundaries

    # Constants
    x_fixed = rob_dict['pick_pos_base'][0]['x']
    frames_lim = 10 # Max frames object must be tracked to start pick & place
    # Predefine packet z and x offsets with robot speed of 55%.
    # Index corresponds to type of packet.
    pack_depths = [10.0, 3.0, 5.0, 5.0] # List of z positions at pick
    pack_x_offsets = [50.0, 180.0, 130.0, 130.0] # List of x positions at pick

    # Variables
    pick_list = [] # List for items ready to be picked
    warn_count = 0 # Counter for markers out of frame or moving
    frame_count = 1 # Counter of frames for homography update
    text_size = 1
    gripper = False
    conv_left = False # Conveyor heading left enable
    conv_right = False # Conveyor heading right enable
    homography = None # Homography matrix
    state = "READY" # Robot state variable

    grip_time_offset = 900

    while True:
        # Start timer for FPS estimation
        start_time = time.time()

        # Read data dict from PLC server
        try:
            rob_stopped = info_dict['rob_stopped']
            stop_active = info_dict['stop_active']
            prog_done = info_dict['prog_done']
            encoder_vel = info_dict['encoder_vel']
            pos = info_dict['pos']
        except:
            continue

        # Read encoder dict from PLC server
        encoder_pos = encoder_pos_m.value
        if encoder_pos is None:
            continue

        # Get frames from realsense.
        success, depth_frame, rgb_frame, colorized_depth = dc.get_aligned_frame()
        if not success:
            continue

        frame_height, frame_width, frame_channel_count = rgb_frame.shape
        text_size = (frame_height / 1000)
        image_frame = rgb_frame.copy()

        # Draw HSV mask over screen if enabled
        if show_hsv_mask and DETECTOR_TYPE == 'hsv':
            image_frame = pack_detect.draw_hsv_mask(image_frame)

        try:
            # Try to detect tags in rgb frame.
            image_frame = apriltag.detect_tags(rgb_frame, image_frame = image_frame)

            # Update homography on first frame of 500 frames.
            if frame_count == 1:
                homography = apriltag.compute_homog()
                print('[INFO]: Homography matrix updated.')

            # If recieving homography matrix as np array.
            is_type_np = isinstance(homography, np.ndarray)
            is_marker_detect = is_type_np or homography == None

            if DETECTOR_TYPE == 'hsv':
                pack_detect.set_homography(homography)

            # Reset not detected tags warning.
            if is_marker_detect:
                warn_count = 0
        
        # Triggered when no markers are in the frame.
        except Exception as e:
            warn_count += 1
            # Print warning only once.
            if warn_count == 1:
                print("[INFO]: Markers out of frame or moving.")
            pass
        
        # Detect packets using neural network
        if DETECTOR_TYPE == 'deep_1':
            image_frame, detected_packets = pack_detect.deep_pack_obj_detector(rgb_frame, 
                                                                               depth_frame,
                                                                               encoder_pos,
                                                                               bnd_box = show_bbox,
                                                                               homography = homography,
                                                                               image_frame = image_frame)
            for packet in detected_packets:
                packet.width = packet.width * frame_width
                packet.height = packet.height * frame_height
        
        # Detect packets using neural network
        elif DETECTOR_TYPE == 'deep_2':
            # TODO Implement new deep detector
            detected_packets = []
            pass
        
        # Detect packets using neural HSV thresholding
        elif DETECTOR_TYPE == 'hsv':
            image_frame, detected_packets = pack_detect.detect_packet_hsv(rgb_frame,
                                                                          encoder_pos,
                                                                          draw_box = show_bbox,
                                                                          image_frame = image_frame)

        # Update tracked packets from detected packets
        labeled_packets = pt.track_items(detected_packets)
        pt.update_item_database(labeled_packets)

        # Update depth frames of tracked packets
        for item in pt.item_database:
            if item.disappeared == 0:
                # Check if packet is far enough from edge
                if item.centroid[0] - item.width / 2 > item.crop_border_px and item.centroid[0] + item.width / 2 < (frame_width - item.crop_border_px):
                    depth_crop = item.get_crop_from_frame(depth_frame)
                    item.add_depth_crop_to_average(depth_crop)

        # Update registered packet list with new packet info
        registered_packets = pt.item_database

        # When detected not empty, objects are being detected
        is_detect = len(detected_packets) != 0
        # When speed of conveyor more than -100 it is moving to the left
        is_conv_mov = encoder_vel < - 100.0
        # Robot ready when programs are fully finished and it isn't moving
        is_rob_ready = prog_done and (rob_stopped or not stop_active)

        # Add to pick list
        # If packets are being tracked.
        if is_detect:
            # If the conveyor is moving to the left direction.
            if is_conv_mov:
                # Increase counter of frames with detections. 
                for packet in registered_packets:
                    packet.track_frame += 1
                    # If counter larger than limit, and packet not already in pick list.
                    if packet.track_frame > frames_lim and not packet.in_pick_list:
                        print("INFO: Add packet ID: {} to pick list".format(str(packet.id)))
                        # Add to pick list.
                        packet.in_pick_list = True
                        pick_list.append(packet) # ? copy

        # ROBOT READY 
        if state == "READY" and is_rob_ready and pick_list and homography is not None:
            encoder_pos = encoder_pos_m.value
            if encoder_pos is None:
                continue
            # Update pick list to current positions
            for packet in pick_list:
                packet.centroid = packet.getCentroidFromEncoder(encoder_pos)
            # Get list of current world x coordinates
            pick_list_positions = np.array([packet.getCentroidInWorldFrame(homography)[0] for packet in pick_list])
            print("DEBUG: Pick distances")
            print(pick_list_positions)
            # If item is too far remove it from list
            is_valid_position = pick_list_positions < 1800 - grip_time_offset  # TODO find position after which it does not pick up - depends on enc_vel and robot speed
            pick_list = np.ndarray.tolist(np.asanyarray(pick_list)[is_valid_position])     
            pick_list_positions = pick_list_positions[is_valid_position]
            # Choose a item for picking
            if pick_list:
                # Chose farthest item on belt
                pick_ID = pick_list_positions.argmax()
                packet_to_pick = pick_list.pop(pick_ID)
                print("INFO: Chose packet ID: {} to pick".format(str(packet.id)))

                # Set positions and Start robot
                packet_x,pick_pos_y = packet_to_pick.getCentroidInWorldFrame(homography)
                # packet_x = x_fixed  # for testing # TODO find offset value from packet
                pick_pos_x = packet_x + grip_time_offset
                angle = packet.angle
                gripper_rot = compute_gripper_rot(angle)
                packet_type = packet.pack_type

                # Set packet depth to fixed value by type
                packet_z = pack_depths[packet_type]

                # Check if y is range of conveyor width and adjust accordingly
                if pick_pos_y < 75.0:
                    pick_pos_y = 75.0

                elif pick_pos_y > 470.0:
                    pick_pos_y = 470.0

                # Check if x is range
                if pick_pos_x < 600.0:
                    pick_pos_x = 600.0

                elif pick_pos_x > 1800.0:
                    pick_pos_x = 1800.0

                prepick_xyz_coords = np.array([pick_pos_x, pick_pos_y, rob_dict['pick_pos_base'][0]['z']])

                # Change end points of robot.   
                trajectory_dict = {
                    'x': pick_pos_x,
                    'y': pick_pos_y,
                    'rot': gripper_rot,
                    'packet_type': packet_type,
                    'x_offset': 0,
                    'pack_z': packet_z
                    }
                control_pipe.send(RcData(RcCommand.CHANGE_TRAJECTORY, trajectory_dict))

                # Start robot program.   #! only once
                control_pipe.send(RcData(RcCommand.START_PROGRAM))
                state = "TO_PREPICK"
                print("state: TO_PREPICK")

        # TO PREPICK
        if state == "TO_PREPICK":
            # check if robot arrived to prepick position
            curr_xyz_coords = np.array(pos[0:3])
            robot_dist = np.linalg.norm(prepick_xyz_coords-curr_xyz_coords)
            if robot_dist > 5: # TODO check value
                state = "WAIT_FOR_PACKET"
                print("state: WAIT_FOR_PACKET")

        # WAIT FOR PACKET
        if state == "WAIT_FOR_PACKET":
            # TODO add return to ready if it misses packet
            encoder_pos = encoder_pos_m.value
            # check encoder and activate robot 
            packet_to_pick.centroid = packet_to_pick.getCentroidFromEncoder(encoder_pos)
            packet_pos_x = packet_to_pick.getCentroidInWorldFrame(homography)[0]
            print("X distance")
            print(pick_pos_x - packet_pos_x)
            # If packet is close enough continue picking operation
            if packet_pos_x > pick_pos_x - 280:
                control_pipe.send(RcData(RcCommand.CONTINUE_PROGRAM))
                state = "PICKING"
                print("state: PICKING")

        if state == "PICKING":
            if is_rob_ready:
                state = "READY"
                print("state: READY")

        # Draw packet info
        for packet in registered_packets:
            if packet.disappeared == 0:
                # Draw packet ID and type
                text_id = "ID {}, Type {}".format(packet.id, packet.type)
                drawText(image_frame, text_id, (packet.centroid_px.x + 10, packet.centroid_px.y), text_size)

                # Draw packet centroid value in pixels
                text_centroid = "X: {}, Y: {} (px)".format(packet.centroid_px.x, packet.centroid_px.y)
                drawText(image_frame, text_centroid, (packet.centroid_px.x + 10, packet.centroid_px.y + int(45 * text_size)), text_size)

                # Draw packet centroid value in milimeters
                text_centroid = "X: {:.2f}, Y: {:.2f} (mm)".format(packet.centroid_mm.x, packet.centroid_mm.y)
                drawText(image_frame, text_centroid, (packet.centroid_px.x + 10, packet.centroid_px.y + int(80 * text_size)), text_size)

                packet_depth_mm = compute_mean_packet_z(packet, pack_depths[packet.type])
                # Draw packet depth value in milimeters
                text_centroid = "Z: {} (mm)".format(packet_depth_mm)
                drawText(image_frame, text_centroid, (packet.centroid_px.x + 10, packet.centroid_px.y + int(115 * text_size)), text_size)

        # Draw packet depth crop to separate frame
        for packet in registered_packets:
            if packet.avg_depth_crop is not None:
                cv2.imshow("Depth Crop", colorizeDepthFrame(packet.avg_depth_crop))
                break

        # Show depth frame overlay.
        if show_depth_map:
            image_frame = cv2.addWeighted(image_frame, 0.8, colorized_depth, 0.3, 0)

        # Show FPS and robot position data
        if show_frame_data:
            # Draw FPS to screen
            text_fps = "FPS: {:.2f}".format(1.0 / (time.time() - start_time))
            drawText(image_frame, text_fps, (10, int(35 * text_size)), text_size)

            # Draw OPCUA data to screen
            text_robot = str(info_dict)
            drawText(image_frame, text_robot, (10, int(75 * text_size)), text_size)

        image_frame = cv2.resize(image_frame, (frame_width // 2, frame_height // 2))

        # Show frames on cv2 window
        cv2.imshow("Frame", image_frame)

        # Increase counter for homography update
        frame_count += 1
        if frame_count == 500:
            frame_count = 1
        
        # Keyboard inputs
        key = cv2.waitKey(1)

        # Toggle gripper
        if key == ord('g'):
            gripper = not gripper
            control_pipe.send(RcData(RcCommand.GRIPPER, gripper))

        # Toggle conveyor in left direction
        if key == ord('n'):
            conv_left = not conv_left
            control_pipe.send(RcData(RcCommand.CONVEYOR_LEFT, conv_left))

        # Toggle conveyor in right direction
        if key == ord('m'):
            conv_right = not conv_right
            control_pipe.send(RcData(RcCommand.CONVEYOR_RIGHT, conv_right))

        # Toggle detected packets bounding box display
        if key == ord('b'):
            show_bbox = not show_bbox
        
        # Toggle depth map overlay
        if key == ord('d'):
            show_depth_map = not show_depth_map

        # Toggle frame data display
        if key == ord('f'):
            show_frame_data = not show_frame_data

        # Toggle HSV mask overlay
        if key == ord ('h'):
            show_hsv_mask = not show_hsv_mask

        # Abort program
        if key == ord('a'):
            control_pipe.send(RcData(RcCommand.ABORT_PROGRAM))
        
        # Continue program
        if key == ord('c'):
            control_pipe.send(RcData(RcCommand.CONTINUE_PROGRAM))
        
        # Stop program
        if key == ord('s'):
            control_pipe.send(RcData(RcCommand.STOP_PROGRAM))

        # Print info
        if key == ord('i'):
            print(is_rob_ready)

        # End main
        if key == 27:
            control_pipe.send(RcData(RcCommand.CLOSE_PROGRAM))
            cv2.destroyAllWindows()
            dc.release()
            break


def program_mode(demos, r_control, r_comm_info, r_comm_encoder):
    """
    Program selection function.

    Parameters:
    rc (object): RobotControl object for program execution.
    rd (object): RobotDemos object containing pick and place demos.

    """
    # Read mode input.
    mode = input()
    # Dictionary with robot positions and robot programs.
    modes_dict = {
        '1':{'dict':Pick_place_dict, 
            'func':demos.main_robot_control_demo},
        '2':{'dict':Pick_place_dict, 
            'func':demos.main_pick_place},
        '3':{'dict':Pick_place_dict_conv_mov, 
            'func':demos.main_pick_place_conveyor_w_point_cloud},
        '4':{'dict':Pick_place_dict_conv_mov, 
            'func':main}
                }

    # If mode is a program key.
    if mode in modes_dict:
        # Set robot positions and robot program
        r_control.rob_dict = modes_dict[mode]['dict']
        robot_prog = modes_dict[mode]['func']

        # If first mode (not threaded) start program
        if mode == '1':
            robot_prog(r_control)

        elif mode == '4':
            with Manager() as manager:
                info_dict = manager.dict()
                encoder_pos = manager.Value('d', None)

                control_pipe_1, control_pipe_2 = Pipe()

                main_proc = Process(target = robot_prog, args = (r_control.rob_dict, paths, files, check_point, info_dict, encoder_pos, control_pipe_1))
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

        # Otherwise start selected threaded program
        else:
            with Manager() as manager:
                info_dict = manager.dict()

                main_proc = Process(target = robot_prog, args = (r_control, paths, files, check_point, info_dict))
                info_server_proc = Process(target = r_comm_info.robot_server, args = (info_dict, ))

                main_proc.start()
                info_server_proc.start()

                # Wait for the main process to end
                main_proc.join()
                info_server_proc.kill()

    # If input is exit, exit python.
    if mode == 'e':
        exit()

    # Return function recursively.
    return program_mode(demos, r_control, r_comm_info, r_comm_encoder)


if __name__ == '__main__':
    # Define model parameters.
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    check_point ='ckpt-3'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    # Define model paths.
    paths = {'ANNOTATION_PATH':os.path.join(
                                            'Tensorflow',
                                            'workspace',
                                            'annotations'),
            'CHECKPOINT_PATH': os.path.join(
                                            'Tensorflow', 
                                            'workspace',
                                            'models',
                                            CUSTOM_MODEL_NAME) 
            }
    files = {'PIPELINE_CONFIG':os.path.join(
                                            'Tensorflow', 
                                            'workspace',
                                            'models',
                                            CUSTOM_MODEL_NAME,
                                            'pipeline.config'),
            'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], 
                                        LABEL_MAP_NAME)
            }

    # Define robot positions dictionaries from json file.
    file = open('robot_positions.json')
    robot_poses = json.load(file)
    Pick_place_dict_conv_mov_slow = robot_poses['Pick_place_dict_conv_mov_slow']
    Pick_place_dict_conv_mov = robot_poses['Pick_place_dict_conv_mov']
    Pick_place_dict = robot_poses['Pick_place_dict']
    
    # Initialize robot demos and robot control objects.
    r_control = RobotControl(None)
    r_comm_info = RobotCommunication()
    r_comm_encoder = RobotCommunication()
    demos = RobotDemos(paths, files, check_point)

    # Show message about robot programs.
    print('Select pick and place mode: \n'+
    '1 : Pick and place with static conveyor and hand gestures\n'+
    '2 : Pick and place with static conveyor and multithreading\n'+
    '3 : Pick and place with moving conveyor, point cloud and multithreading\n'+
    '4 : Main Pick and place\n'+
    'e : To exit program\n')

    # Start program mode selection.
    program_mode(demos, r_control, r_comm_info, r_comm_encoder)
