import json

import numpy as np
import pyrealsense2 as rs


class IntelConfig:
    """
    Class for loading json config file into depth camera.
    """

    def __init__(self):
        """
        IntelConfig object constructor.

        Args:
            config_path (str): Path to the config file.
        """

        self.DS5_product_ids = [
            "0AD1",
            "0AD2",
            "0AD3",
            "0AD4",
            "0AD5",
            "0AF6",
            "0AFE",
            "0AFF",
            "0B00",
            "0B01",
            "0B03",
            "0B07",
            "0B3A",
            "0B5C",
        ]

    def find_device_that_supports_advanced_mode(self) -> rs.device:
        """
        Searches devices connected to the PC for one compatible with advanced mode.

        Returns:
            rs.device: RealSense device which supports advanced mode.
        """

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            if (
                dev.supports(rs.camera_info.product_id)
                and dev.supports(rs.camera_info.name)
                and str(dev.get_info(rs.camera_info.product_id)) in self.DS5_product_ids
            ):
                print(
                    "[INFO] Found device that supports advanced mode:",
                    dev.get_info(rs.camera_info.name),
                )
                return dev

        raise Exception(
            "[ERROR] No RealSense camera that supports advanced mode was found"
        )

    def load_config(self, config_path: str):
        """
        Loads json config file into the camera.

        Args:
            config_path (str): Path to the config file.
        """

        # Open camera in advanced mode
        dev = self.find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)

        # Read configuration JSON file as string and print it to console
        # serialized_string = advnc_mode.serialize_json()
        # print(serialized_string)

        # Write configuration file to camera
        with open(config_path) as file:
            data = json.load(file)
        json_string = str(data).replace("'", '"')
        advnc_mode.load_json(json_string)
        print("[INFO] Loaded RealSense camera config from file:", config_path)


class DepthCamera:
    """
    Class for connecting to and reading images from Inteal RealSense depth camera.
    """

    def __init__(
        self,
        config_path: str = None,
    ):
        """
        IntelConfig object constructor.

        Args:
            config_path (str): Path to the config file.
        """

        # Check if any RealSense camera is connected
        ctx = rs.context()
        devices = ctx.query_devices()
        is_camera_connected = len(devices) >= 1

        if not is_camera_connected:
            raise Exception("[ERROR] No RealSense camera was found")

        if config_path is not None:
            # If config file path was given, load camera config from provided file
            self.ic = IntelConfig()
            self.ic.load_config(config_path)
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Setup RGB and Depth stream resolution, format and FPS
        # Maximal supported Depth stream resolution of D435 camera is 1280 x 720
        # Maximal supported RGB stream resolution of D435 camera is 1920 x 1080
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)

        # Create object for aligning depth frame to RGB frame, so that they have equal resolution
        self.align = rs.align(rs.stream.color)

        # Create object for filling missing depth pixels where the sensor was not able to detect depth
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 2)

        # Create object for colorizing depth frames
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 0)

        # Start video stream
        self.profile = self.pipeline.start(self.config)

        profile = self.profile.get_stream(rs.stream.color)
        self.intr = profile.as_video_stream_profile().get_intrinsics()

        # Code for getting camera intrinsic parameters
        # self.stream_profile_color = self.profile.get_stream(rs.stream.color, 0)
        # self.stream_profile_depth = self.profile.get_stream(rs.stream.depth, 0)
        # self.video_stream_profile = self.stream_profile_color.as_video_stream_profile()
        # ; = self.video_stream_profile.get_intrinsics()
        # print("fx =", intrinsics.fx, "; fy =", intrinsics.fy)
        # print("ppx =", intrinsics.ppx, "; ppy =", intrinsics.ppy)
        # print("height =", intrinsics.height, "; width =", intrinsics.width)
        # print("coeffs =", intrinsics.coeffs)
        # print("model =", intrinsics.model)

    def get_frames(self) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads and processes frames from connected camera.

        Returns:
            bool: True if frame was succesfully read
            float: Time in milliseconds when the frame was captured
            np.ndarray: RGB frame
            np.ndarray: Depth frame
            np.ndarray: Colorized depth frame
        """

        # Reads RGB and depth frames and resize them to same resolution
        frameset = self.pipeline.wait_for_frames()
        frameset = self.align.process(frameset)

        # Extract RGB and depth frames from frameset
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        frame_timestamp = frameset.get_timestamp()

        if not depth_frame or not color_frame:
            return False, None, None, None

        # Apply hole filling filter
        depth_frame = self.hole_filling.process(depth_frame)

        colorized_depth_frame = np.asanyarray(
            self.colorizer.colorize(depth_frame).get_data()
        )
        depth_frame = np.asanyarray(depth_frame.get_data())
        color_frame = np.asanyarray(color_frame.get_data())

        return True, frame_timestamp, depth_frame, color_frame, colorized_depth_frame

    def get_raw_depth_frame(self):
        frameset = self.pipeline.wait_for_frames()
        frameset = self.align.process(frameset)
        return frameset.get_depth_frame()

    def pixel_to_3d_point(self, pixel: list[int, int], depth_frame: rs.depth_frame):
        dist = depth_frame.get_distance(pixel[0], pixel[1])
        coordinates_3d = rs.rs2_deproject_pixel_to_point(
            self.intr, [pixel[0], pixel[1]], dist
        )
        coordinates_3d = [coordinates_3d[0], coordinates_3d[1], coordinates_3d[2]]

        return coordinates_3d
    
    def release(self):
        """
        Disconnects the camera.
        """

        self.pipeline.stop()
