import cv2
import time
import numpy as np

# Class used for calibrating encoder
# 1. Call saveStartValues() to save centroid X position and encoder value
# 2. Move packet
# 3. Call saveEndValues() to save different centroid X position and encoder value
# 4. Call calculateConstant() to get conversion constant from encoder to centroid X position
class EncoderCalibration:
    def __init__(self):
        self.real_pos_start = 0
        self.real_pos_end = 1
        self.encoder_pos_start = 0
        self.encoder_pos_end = 1

        self.constant = 1

    def saveStartValues(self, real_pos, encoder_pos):
        self.real_pos_start = real_pos
        self.encoder_pos_start = encoder_pos
        print("Saved start values")

    def saveEndValues(self, real_pos, encoder_pos):
        self.real_pos_end = real_pos
        self.encoder_pos_end = encoder_pos
        print("Saved end values")

    def calculateConstant(self):
        self.constant = (self.real_pos_end - self.real_pos_start) / (self.encoder_pos_end - self.encoder_pos_start)
        return self.constant

    def printConstant(self):
        print("Encoder calibration constant: {}".format(self.constant))

from robot_cell.detection.apriltag_detection import ProcessingApriltag
from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.threshold_detector import ThresholdDetector
from robot_cell.control.robot_communication import RobotCommunication


if __name__ == "__main__":
    ec = EncoderCalibration()

    dc = DepthCamera()

    pack_detect = ThresholdDetector()

    apriltag = ProcessingApriltag()
    apriltag.load_world_points('config/conveyor_points.json')

    rc = RobotCommunication()

    rc.connect_OPCUA_server()
    rc.get_nodes()
    
    time.sleep(0.5)

    while True:
        success, depth_frame, rgb_frame, colorized_depth = dc.get_frames()
        if not success:
            continue

        frame_height, frame_width, frame_channel_count = rgb_frame.shape
        image_frame = rgb_frame.copy()

        apriltag.detect_tags(rgb_frame)
        homography = apriltag.compute_homog()
        if not isinstance(homography, np.ndarray):
            continue
        pack_detect.set_homography(homography)
        image_frame = apriltag.draw_tags(image_frame)

        encoder_pos = round(rc.Encoder_Pos.get_value(), 2)

        image_frame, detected_packets, _ = pack_detect.detect_packet_hsv(
            rgb_frame,
            encoder_pos,
            draw_box=True,
            image_frame=image_frame,
        )

        # cv2.imshow("Frame", cv2.resize(image_frame, (frame_width // 2, frame_height // 2)))
        cv2.imshow("Frame", image_frame)
        
        key = cv2.waitKey(1)

        if key == 27:
            dc.release()
            rc.client.disconnect()
            cv2.destroyAllWindows()
            break

        # Info
        if key == ord('i'):
            print("Encoder value:", encoder_pos)

        # Start encoder calibration
        if key == ord('1'):
            # Use centroid of first packet
            for packet in detected_packets:
                ec.saveStartValues(packet.centroid_px[0], encoder_pos)
                break

        # Finish encoder calibration
        if key == ord('2'):
            # Use centroid of first packet
            for packet in detected_packets:
                ec.saveEndValues(packet.centroid_px[0], encoder_pos)
                break
            ec.calculateConstant()
            ec.printConstant()
