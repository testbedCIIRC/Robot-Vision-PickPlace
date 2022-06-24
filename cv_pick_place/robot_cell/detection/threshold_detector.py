import numpy as np
import cv2

from robot_cell.packet.packet_object import Packet

ignore_region_vertical = 60
ignore_region_horizontal = 10

class ThresholdDetector:
    def __init__(self):
        self.detected_objects = []
        self.homography_matrix = None
        self.homography_determinant = None
        
    def set_homography(self, homography_matrix):
        self.homography_matrix = homography_matrix
        self.homography_determinant = np.linalg.det(homography_matrix[0:2, 0:2])
        
    def get_packet_from_contour(self, contour, type, depth_frame, encoder_pos):
        rect = cv2.minAreaRect(contour)
        centroid = (int(rect[0][0]), int(rect[0][1]))
        box = np.int0(cv2.boxPoints(rect))
        angle = int(rect[2])
        x, y, w, h = cv2.boundingRect(contour)

        packet = Packet(box = box, 
                        pack_type = type,
                        centroid = centroid,
                        centroid_depth = depth_frame[centroid[1]][centroid[0]], 
                        angle = angle,
                        ymin = y, ymax = y + w, 
                        xmin = x, xmax = x + h, 
                        width = w, height = h, 
                        encoder_position = encoder_pos)
        
        return packet
        
    def detect_packet_hsv(self, rgb_frame, depth_frame, encoder_pos):
        self.detected_objects = []
        
        if self.homography_determinant is None:
            print("[WARINING] ObjectDetector: No homography matrix set")
            return rgb_frame, self.detected_objects
        
        frame_height = rgb_frame.shape[0]
        frame_width = rgb_frame.shape[1]

        image_frame = rgb_frame.copy()
        
        # Get binary mask
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
        
        # blue_lower = np.array([60, 35, 140])
        # blue_upper = np.array([180, 255, 255])
        # blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
        
        brown_lower = np.array([5, 20, 70])
        brown_upper = np.array([35, 255, 255])
        brown_mask = cv2.inRange(hsv_frame, brown_lower, brown_upper)
        brown_mask[:ignore_region_vertical, :] = 0
        brown_mask[(frame_height - ignore_region_vertical):, :] = 0
        
        white_lower = np.array([40, 0, 90])
        white_upper = np.array([140, 255, 255])
        white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)
        white_mask[:ignore_region_vertical, :] = 0
        white_mask[(frame_height - ignore_region_vertical):, :] = 0

        brown_contour_list, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_contour_list, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect white packets from white binary mask contours
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
            packet = self.get_packet_from_contour(contour, object_type, depth_frame, encoder_pos)
            
            # Check for squareness
            side_ratio = packet.width / packet.height
            if not 1.1 > side_ratio > 0.9:
                continue

            # Check if packet is far enough from edge
            if packet.centroid[0] - packet.width / 2 < ignore_region_horizontal or packet.centroid[0] + packet.width/2 > (frame_width - ignore_region_horizontal):
                continue

            cv2.rectangle(image_frame, 
                      (packet.centroid[0] - int(packet.width / 2), packet.centroid[1] - int(packet.height / 2)), 
                      (packet.centroid[0] + int(packet.width / 2), packet.centroid[1] + int(packet.height / 2)), 
                      (255, 0, 0), 2, lineType=cv2.LINE_AA)

            cv2.drawContours(image_frame, 
                            [packet.box], 
                            -1, 
                            (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.drawMarker(image_frame, 
                        packet.centroid, 
                        (0, 0, 255), cv2.MARKER_CROSS, 10, cv2.LINE_4)
            
            self.detected_objects.append(packet)

        # Detect brown packets from brown binary mask contours
        for contour in brown_contour_list:
            area_cm2 = abs(cv2.contourArea(contour) * self.homography_determinant)
            object_type = 0
            
            if 165 > area_cm2 > 145:
                object_type = 3
            else:
                continue
            
            # Get detected packet parameters
            packet = self.get_packet_from_contour(contour, object_type)

            # Check for squareness
            side_ratio = packet.width / packet.height
            if not 1.1 > side_ratio > 0.9:
                continue

            # Check if packet is far enough from edge
            if packet.centroid[0] - packet.width / 2 < ignore_region_horizontal or packet.centroid[0] + packet.width/2 > (frame_width - ignore_region_horizontal):
                continue

            cv2.rectangle(image_frame, 
                      (packet.centroid[0] - int(packet.width / 2), packet.centroid[1] - int(packet.height / 2)), 
                      (packet.centroid[0] + int(packet.width / 2), packet.centroid[1] + int(packet.height / 2)), 
                      (255, 0, 0), 2, lineType=cv2.LINE_AA)

            cv2.drawContours(image_frame, 
                            [packet.box], 
                            -1, 
                            (0, 255, 0), 2, lineType=cv2.LINE_AA)

            cv2.drawMarker(image_frame, 
                        packet.centroid, 
                        (0, 0, 255), cv2.MARKER_CROSS, 10, cv2.LINE_4)
            
            self.detected_objects.append(packet)

        return image_frame, self.detected_objects
