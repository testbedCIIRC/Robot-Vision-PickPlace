import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        # Start streaming
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.enable_auto_exposure, False)
        rgb_sensor = profile.get_device().query_sensors()[1]
        rgb_sensor.set_option(rs.option.enable_auto_exposure, False)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, 2)
        depth_frame = hole_filling.process(depth_frame)

        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, 0)

        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None, None
        return True, depth_image, color_image, colorized_depth

    def release(self):
        self.pipeline.stop()