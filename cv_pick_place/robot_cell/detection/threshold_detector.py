from struct import pack
import numpy as np
import cv2

from robot_cell.packet.item_object import Item
from robot_cell.packet.packet_object import Packet
from robot_cell.functions import *

class ThresholdDetector:
    def __init__(self, ignore_vertical_px = 60, ignore_horizontal_px = 10, max_ratio_error = 0.1,
                       white_lower = [40, 0, 90], white_upper = [140, 255, 255],
                       brown_lower = [5, 20, 70], brown_upper = [35, 255, 255]):
        self.detected_objects = []
        self.homography_matrix = None
        self.homography_determinant = None

        self.ignore_vertical_px = ignore_vertical_px
        self.ignore_horizontal_px = ignore_horizontal_px

        self.max_ratio_error = max_ratio_error

        self.white_lower = np.array([white_lower])
        self.white_upper = np.array([white_upper])

        self.brown_lower = np.array([brown_lower])
        self.brown_upper = np.array([brown_upper])
        
    def set_homography(self, homography_matrix):
        self.homography_matrix = homography_matrix
        self.homography_determinant = np.linalg.det(homography_matrix[0:2, 0:2])
        
    def get_packet_from_contour(self, contour, type, encoder_pos):
        rectangle = cv2.minAreaRect(contour)
        centroid = (int(rectangle[0][0]), int(rectangle[0][1]))
        box = np.int0(cv2.boxPoints(rectangle))
        angle = int(rectangle[2])
        x, y, w, h = cv2.boundingRect(contour)

        packet = Packet(box = box, 
                        pack_type = type,
                        centroid = centroid,
                        angle = angle,
                        ymin = y, ymax = y + w, 
                        xmin = x, xmax = x + h, 
                        width = w, height = h, 
                        encoder_position = encoder_pos)

        packet.set_type(type)
        packet.set_centroid(centroid[0], centroid[1], self.homography_matrix, encoder_pos)
        packet.set_bounding_size(w, h, self.homography_matrix)
        packet.add_angle_to_average(angle)

        return packet

    def draw_packet_info(self, image_frame, packet, encoder_position, draw_box = True, text_size = 1):
        if draw_box:
            # Draw bounding rectangle
            cv2.rectangle(image_frame, 
                          (packet.centroid[0] - int(packet.width / 2), packet.centroid[1] - int(packet.height / 2)), 
                          (packet.centroid[0] + int(packet.width / 2), packet.centroid[1] + int(packet.height / 2)), 
                          (255, 0, 0), 2, lineType=cv2.LINE_AA)

            # Draw item contours
            cv2.drawContours(image_frame, 
                             [packet.box], 
                             -1, 
                             (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Draw centroid
        cv2.drawMarker(image_frame, 
                       packet.centroid, 
                       (0, 0, 255), cv2.MARKER_CROSS, 10, cv2.LINE_4)

        # Draw centroid estimated with encoder position
        cv2.drawMarker(image_frame, 
                       packet.getCentroidFromEncoder(encoder_position), 
                       (255, 255, 0), cv2.MARKER_CROSS, 10, cv2.LINE_4)

        return image_frame
        
    def detect_packet_hsv(self, rgb_frame, depth_frame, encoder_position, draw_box = True, text_size = 1, image_frame = None):
        self.detected_objects = []
        
        if self.homography_determinant is None:
            print("[WARINING] ObjectDetector: No homography matrix set")
            return image_frame, self.detected_objects

        if image_frame is None or not image_frame.shape == rgb_frame.shape:
            image_frame = rgb_frame.copy()
        
        frame_height = rgb_frame.shape[0]
        frame_width = rgb_frame.shape[1]
        
        # Get binary mask
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
        
        white_mask = cv2.inRange(hsv_frame, self.white_lower, self.white_upper)
        white_mask[:self.ignore_vertical_px, :] = 0
        white_mask[(frame_height - self.ignore_vertical_px):, :] = 0

        brown_mask = cv2.inRange(hsv_frame, self.brown_lower, self.brown_upper)
        brown_mask[:self.ignore_vertical_px, :] = 0
        brown_mask[(frame_height - self.ignore_vertical_px):, :] = 0

        white_contour_list, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brown_contour_list, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect WHITE packets from white binary mask contours
        for contour in white_contour_list:
            area_cm2 = abs(cv2.contourArea(contour) * self.homography_determinant)
            object_type = 0
            
            if 110 > area_cm2 > 90:
                object_type = 1
            elif 180 > area_cm2 > 160:
                object_type = 2
            elif 380 > area_cm2 > 350:
                object_type = 0
            else:
                continue
            
            # Get detected packet parameters
            packet = self.get_packet_from_contour(contour, object_type, encoder_position)
            
            # Check for squareness
            side_ratio = packet.width / packet.height
            if not (1 + self.max_ratio_error) > side_ratio > (1 - self.max_ratio_error):
                continue

            # Check if packet is far enough from edge
            if packet.centroid[0] - packet.width / 2 < self.ignore_horizontal_px or packet.centroid[0] + packet.width / 2 > (frame_width - self.ignore_horizontal_px):
                continue

            image_frame = self.draw_packet_info(image_frame, packet, encoder_position, draw_box, text_size)
            
            self.detected_objects.append(packet)

        # Detect BROWN packets from brown binary mask contours
        for contour in brown_contour_list:
            area_cm2 = abs(cv2.contourArea(contour) * self.homography_determinant)
            object_type = 0
            
            if 165 > area_cm2 > 145:
                object_type = 3
            else:
                continue
            
            # Get detected packet parameters
            packet = self.get_packet_from_contour(contour, object_type, encoder_position)

            # Check for squareness
            side_ratio = packet.width / packet.height
            if not (1 + self.max_ratio_error) > side_ratio > (1 - self.max_ratio_error):
                continue

            # Check if packet is far enough from edge
            if packet.centroid[0] - packet.width / 2 < self.ignore_horizontal_px or packet.centroid[0] + packet.width / 2 > (frame_width - self.ignore_horizontal_px):
                continue
            
            image_frame = self.draw_packet_info(image_frame, packet, encoder_position, draw_box, text_size)

            self.detected_objects.append(packet)

        return image_frame, self.detected_objects
