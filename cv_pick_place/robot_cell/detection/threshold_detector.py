import numpy as np
import cv2

from robot_cell.packet.packet_object import Packet

# Workstation with json camera config
# White:
# Lower HSV bounds: [60, 0, 85]
# Upper HSV bounds: [179, 255, 255]
# Frame bounds: 133
# Brown:
# Lower HSV bounds: [0, 33, 57]
# Upper HSV bounds: [60, 255, 178]
# Frame bounds: 133

class ThresholdDetector:
    """
    Detects white and brown packets in image using HSV threasholding.
    """

    def __init__(self,
                 ignore_vertical_px: int = 60,
                 ignore_horizontal_px: int = 10,
                 max_ratio_error: float = 0.1,
                 white_lower: list[int] = [40, 0, 90],
                 white_upper: list[int] = [140, 255, 255],
                 brown_lower: list[int] = [5, 20, 70],
                 brown_upper: list[int] = [35, 255, 255]) -> None:
        """
        ThresholdDetector object constructor.

        Args:
            ignore_vertical_px (int): Number of rows of pixels ignored from top and bottom of the image frame.
            ignore_horizontal_px (int): Number of columns of pixels ignored from left and right of the image frame.
            max_ratio_error (float): Checking squareness of the packet allows the ratio of packet sides to be off by this ammount.
            white_lower (list[int]): List of 3 values representing Hue, Saturation, Value bottom threshold for white color.
            white_upper (list[int]): List of 3 values representing Hue, Saturation, Value top threshold for white color.
            brown_lower (list[int]): List of 3 values representing Hue, Saturation, Value bottom threshold for brown color.
            brown_upper (list[int]): List of 3 values representing Hue, Saturation, Value top threshold for brown color.
        """

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
        
    def set_homography(self, homography_matrix: np.ndarray) -> None:
        """
        Sets the homography matrix and calculates its determinant.

        Args:
            homography_matrix(np.ndarray): Homography matrix.
        """

        self.homography_matrix = homography_matrix
        self.homography_determinant = np.linalg.det(homography_matrix[0:2, 0:2])
        
    def get_packet_from_contour(self, contour: np.array, type: int, encoder_pos: float) -> Packet:
        """
        Creates Packet object from a contour.

        Args:
            contour (np.array): Array of x, y coordinates making up the contour of some area.
            type (int): Type of the packet.
            encoder_pos (float): Position of the encoder.

        Returns:
            Packet: Created Packet object
        """

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

    def draw_packet_info(self,
                         image_frame: np.ndarray,
                         packet: Packet,
                         encoder_position: float,
                         draw_box: bool = True) -> np.ndarray:
        """
        Draws information about a packet into image.

        Args:
            image_frame (np.ndarray): Image into which the information should be drawn.
            packet (Packet): Packet object whose information should be drawn.
            encoder_position (float): Position of the encoder.
            draw_box (bool): If bounding and min area boxes should be drawn.

        Returns:
            np.ndarray: Image frame with information drawn into it.
        """
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
                       (0, 0, 255), cv2.MARKER_CROSS, 20, cv2.LINE_4)

        return image_frame
        
    def detect_packet_hsv(self,
                          rgb_frame: np.ndarray,
                          encoder_position: float,
                          draw_box: bool = True,
                          image_frame: np.ndarray = None) -> tuple[np.ndarray, list[Packet], np.ndarray]:
        """
        Detects packets using HSV thresholding in an image.

        Args:
            rgb_frame (np.ndarray): RGB frame in which packets should be detected.
            encoder_position (float): Position of the encoder.
            draw_box (bool): If bounding and min area boxes should be drawn.
            image_frame (np.ndarray): Image frame into which information should be drawn.

        Returns:
            np.ndarray: Image frame with information drawn into it.
            list[Packet]: List of detected packets.
            np.ndarray: Binary detection mask.
        """

        self.detected_objects = []
        
        if self.homography_determinant is None:
            print("[WARINING] ObjectDetector: No homography matrix set")
            return image_frame, self.detected_objects, None

        if image_frame is None or not image_frame.shape == rgb_frame.shape:
            image_frame = rgb_frame.copy()
        
        frame_height = rgb_frame.shape[0]
        frame_width = rgb_frame.shape[1]

        mask = np.zeros((frame_height, frame_width))

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

            image_frame = self.draw_packet_info(image_frame, packet, encoder_position, draw_box)
            mask = cv2.drawContours(mask, [contour], -1, (1),thickness=cv2.FILLED)

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
            
            image_frame = self.draw_packet_info(image_frame, packet, encoder_position, draw_box)
            mask = cv2.drawContours(mask, [contour], -1, (1),thickness=cv2.FILLED)

            self.detected_objects.append(packet)

        bin_mask = mask.astype(bool)
        return image_frame, self.detected_objects, bin_mask

    def draw_hsv_mask(self, image_frame: np.ndarray) -> np.ndarray:
        """
        Draws binary HSV mask into image frame.

        Args:
            image_frame (np.ndarray): Image frame into which the mask should be drawn.

        Returns:
            np.ndarray: Image frame with information drawn into it.
        """
        frame_height = image_frame.shape[0]
        frame_width = image_frame.shape[1]
        
        # Get binary mask
        hsv_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)
        
        white_mask = cv2.inRange(hsv_frame, self.white_lower, self.white_upper)
        white_mask[:self.ignore_vertical_px, :] = 0
        white_mask[(frame_height - self.ignore_vertical_px):, :] = 0

        brown_mask = cv2.inRange(hsv_frame, self.brown_lower, self.brown_upper)
        brown_mask[:self.ignore_vertical_px, :] = 0
        brown_mask[(frame_height - self.ignore_vertical_px):, :] = 0

        mask = cv2.bitwise_or(white_mask, brown_mask)

        image_frame = cv2.bitwise_and(image_frame, image_frame, mask=mask)

        return image_frame
