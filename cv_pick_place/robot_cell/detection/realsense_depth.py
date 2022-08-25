import json
import time

import cv2
import numpy as np
import pyrealsense2 as rs


class IntelConfig:
    """
    Class for loading json config file into depth camera.
    """

    def __init__(self, config_path: str):
        """
        IntelConfig object constructor.

        Args:
            config_path (str): Path to the config file.
        """

        self.DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]
        self.config_path = config_path

    def find_device_that_supports_advanced_mode(self):
        """
        Searches devices connected to the PC for one compatible with advanced mode.

        Returns:
            rs.device: RealSense device.
        """

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(
                    dev.get_info(rs.camera_info.product_id)) in self.DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No device that supports advanced mode was found")

    def open_camera(self):
        """
        Loads json config file into the camera.

        Returns:
            rs.pipeline: Pipeline object which can be used to reads frames from camera.
        """

        # Open camera in advanced mode
        dev = self.find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)

        # read configuration file exported from RealSense Viewer
        # Configuration file is edit of edge detector from intelsense sdk website which work on the parameter
        # texture count and difference

        # serialized_string = advnc_mode.serialize_json()
        # print("Controls as JSON: \n", serialized_string)

        with open(self.config_path) as f:
            data = json.load(f)

        json_string = str(data).replace("'", '\"')
        advnc_mode.load_json(json_string)

        # Configure depth and color streams
        ctx = rs.context()
        pipeline = rs.pipeline(ctx)
        return pipeline


class DepthCamera:
    """
    Class for connecting to and reading images from Inteal RealSense depth camera.
    """

    def __init__(self, config_path = 'D435_camera_config_defaults.json', recording_path = 'recording_2022_07_13.npy', recording_fps = 5):
        """
        IntelConfig object constructor.

        Args:
            config_path (str): Path to the config file.
            recording_path (str): Path to a recording file, in case no camera is detected.
            recording_fps (int): FPS of the recording.
        """
        
        # Check if any RealSense camera is connected
        ctx = rs.context()
        devices = ctx.query_devices()
        self.is_camera_connected = len(devices) >= 1

        # If cameras were detected, start video stream
        if self.is_camera_connected:
            # Configure depth and color streams
            if config_path is None:
                self.pipeline = rs.pipeline()
            else:
                self.ic = IntelConfig(config_path)
                self.pipeline = self.ic.open_camera()
            self.config = rs.config()

            # Get device product line for setting a supporting resolution
            # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            # pipeline_profile = config.resolve(pipeline_wrapper)
            # device = pipeline_profile.get_device()
            # device_product_line = str(device.get_info(rs.camera_info.product_line))

            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

            # Create frame alignment filter
            self.align = rs.align(rs.stream.color)

            # Create hole filling filter
            self.hole_filling = rs.hole_filling_filter()
            self.hole_filling.set_option(rs.option.holes_fill, 2)

            # Create temporal filter
            self.temporal_filter = rs.temporal_filter()

            # Create colorizer
            self.clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 5))

            # Start streaming
            self.profile = self.pipeline.start(self.config)
            self.device = self.profile.get_device()
            depth_sensor = self.device.query_sensors()[0]
            depth_sensor.set_option(rs.option.enable_auto_exposure, False)
            rgb_sensor = self.device.query_sensors()[1]
            rgb_sensor.set_option(rs.option.enable_auto_exposure, False)

        # If no camera was detected, open file with a recording
        else:
            with open(recording_path, 'rb') as f:
                self.recording = np.load(f)

            self.frame_count = self.recording.shape[3]
            self.frame_index = 0
            
            # Delay between frames in nanoseconds
            self.frame_delay_ns = (10.0 ** 9) // recording_fps
            self.last_time_ns = time.time_ns()

            # CLAHE for colorizing depth frames
            self.clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 5))

    def get_frame(self):
        """
        Reads and processes frame from connected camera.

        Returns:
            bool: True if frame was succesfully read
            np.ndarray: RGB frame
            np.ndarray: Depth frame
            np.ndarray: Colorized depth frame
        """

        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        depth_frame = self.hole_filling.process(depth_frame)
        # depth_frame = self.temporal.process(depth_frame)

        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, 0)

        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None, None
        return True, depth_image, color_image, colorized_depth

    def get_aligned_frame(self):
        """
        Gets a frame from either a camera or a recording, if no camera was found.

        Returns:
            bool: True if frame was succesfully read
            np.ndarray: RGB frame
            np.ndarray: Depth frame
            np.ndarray: Colorized depth frame
        """
        if self.is_camera_connected:
            return self.camera_frame_function()
        else:
            return self.recorded_frame_function()

    def camera_frame_function(self):
        """
        Reads and processes frame from connected camera.

        Returns:
            bool: True if frame was succesfully read
            np.ndarray: RGB frame
            np.ndarray: Depth frame
            np.ndarray: Colorized depth frame
        """

        frameset = self.pipeline.wait_for_frames()
        frameset = self.align.process(frameset)

        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None, None

        depth_frame = self.hole_filling.process(depth_frame)
        #depth_frame = self.temporal.process(depth_frame)

        depth_frame = np.asanyarray(depth_frame.get_data())
        color_frame = np.asanyarray(color_frame.get_data())

        colorized_depth_hist = self.clahe.apply(depth_frame.astype(np.uint8))
        colorized_depth_frame = cv2.applyColorMap(colorized_depth_hist, cv2.COLORMAP_JET)

        return True, depth_frame, color_frame, colorized_depth_frame

    def recorded_frame_function(self):
        """
        Reads and processes frame from a recording.

        Returns:
            bool: True if frame was succesfully read
            np.ndarray: RGB frame
            np.ndarray: Depth frame
            np.ndarray: Colorized depth frame
        """

        # Block until time is right
        while time.time_ns() - self.last_time_ns < self.frame_delay_ns:
            time.sleep(0.001)
        
        # Update time
        self.last_time_ns = time.time_ns()
        
        # Read frames from the recording
        rgb_frame = self.recording[:, :, 0:3, self.frame_index].astype(np.uint8)
        depth_frame = self.recording[:, :, 3, self.frame_index].astype(np.uint16)
        
        # Increment frame index
        self.frame_index += 1
        if self.frame_index >= self.frame_count:
            self.frame_index = 0
        
        # Colorize depth frame
        depth_frame_hist = self.clahe.apply(depth_frame.astype(np.uint8))
        colorized_depth_frame = cv2.applyColorMap(depth_frame_hist, cv2.COLORMAP_JET)
        
        return True, depth_frame, rgb_frame, colorized_depth_frame

    def release(self):
        """
        Disconnects the camera.
        """

        if self.is_camera_connected:
            self.pipeline.stop()
