import numpy as np
from math import sqrt
from collections import namedtuple


CM2MM = 10


class Packet:
    """
    Class wrapping relevant information about packets together.
    """

    def __init__(
        self,
        crop_border_px: int = 50,
    ):
        """
        Packet object constructor.

        Args:
            crop_border_px (int): Size of the border in pixels around the packet that is included in the depth crop.
        """

        # NAMED TUPLES
        ##############
        self.PointTuple = namedtuple("Point", ["x", "y"])

        # NEW PACKET PARAMETERS
        #######################
        # ID of the packet for tracking between frames
        self.id = None

        # Type of the packet
        self.type = None

        # X Y centroid value in frame pixels,
        # of the position where the packet last last detected by the camera.
        # X increases from left to right.
        # Y increases from top to bottom.
        self.centroid_px = None

        # Width of horizontally aligned bounding box in pixels
        self.bounding_width_px = None
        # Height of horizontally aligned bounding box in pixels
        self.bounding_height_px = None

        # A 3x3 homography matrix, which corresponds to the transformation
        # from the pixels coordinates of the image frame where the packet was detected
        # into real-world coordinates
        self.homography_matrix = None

        # Time in milliseconds, when the packet data was last updated
        # It should correspond with the data capture timestamp,
        # that is the time when the data used for packet parameters was captured.
        # TODO: Implement packet timestamps
        self.timestamp_ms = None

        # Binary mask for the depth detection
        # TODO: Find out how mask is used
        self.mask = None

        # Ammount of extra pixels around each depth crop
        self.crop_border_px = crop_border_px
        # Number of averaged depth crops
        self.num_avg_depths = 0
        # Numpy array with average depth value for the packet
        self.avg_depth_crop = None

        # The angle gives the rotation of packet contours from horizontal line in the frame
        # Number of averaged angles
        self.num_avg_angles = 0
        # Number with average angle of the packet
        self.avg_angle_deg = None

        # Last detected position of packet centroid in pixels
        self.centroid_base_px = None

        # Last detected position of encoder in pixels
        self.encoder_base_position = None

        # Number of frames the packet has been tracked for
        self.track_frame = 0

        # Indicates if the packet is marked to be the next packet to be sorted
        self.in_pick_list = False

        # Number of frames the packet has disappeared for
        # TODO: Find out how "disappeared" counter is used
        self.disappeared = 0

        # OBSOLETE PACKET PARAMETERS
        ############################

        # Width and height of packet bounding box
        self.width = 0
        self.height = 0

    def set_id(self, packet_id: int) -> None:
        """
        Sets packet ID.

        Args:
            packet_id (int): Unique integer greater then or equal to 0.
        """

        if packet_id < 0:
            print(
                f"[WARNING] Tried to set packet ID to {packet_id} (Should be greater than or equal to 0)"
            )
            return

        self.id = packet_id

    def set_type(self, packet_type: int) -> None:
        """
        Sets packet type.

        Args:
            packet_type (int): Integer representing type of the packet.
        """

        self.type = packet_type

    def set_centroid(self, x: int, y: int) -> None:
        """
        Sets packet centroid.

        Args:
            x (int): X coordinate of centroid in pixels.
            y (int): Y coordinate of centroid in pixels.
        """

        self.centroid_px = self.PointTuple(x, y)

    def set_homography_matrix(self, homography_matrix: np.ndarray) -> None:
        """
        Sets a homography matrix corresponding to the transformation between pixels and centimeters.

        Args:
            homography_matrix (np.ndarray): 3x3 homography matrix.
        """

        self.homography_matrix = homography_matrix

    def set_base_encoder_position(self, encoder_position: float) -> None:
        """
        Sets packet base encoder position and associated centroid position.

        Args:
            encoder_pos (float): Position of encoder.
        """

        self.centroid_base_px = self.centroid_px
        self.encoder_base_position = encoder_position

    def update_timestamp(self, timestamp: float) -> None:
        """
        Updates packet timestamp variable.
        The time should correspond to the time when the data used for packet parameter computation was captured.

        Args:
            timestamp (float): Time in milliseconds.
        """

        self.timestamp_ms = timestamp

    def set_bounding_size(self, width: int, height: int) -> None:
        """
        Sets width and height of packet bounding box.

        Args:
            width (int): Width of the packet in pixels.
            height (int): Height of the packet in pixels.
        """

        if width <= 0 or height <= 0:
            print(
                "[WARNING] Tried to set packet WIDTH and HEIGHT to ({}, {}) (Should be greater than 0)".format(
                    width, height
                )
            )
            return

        self.bounding_width_px = width
        self.bounding_height_px = height

        # OBSOLETE PARAMS
        # TODO: Replace packet width and height with bounding_width_px and bounding_height_px
        self.width = width
        self.height = height

    def set_mask(self, mask: tuple[int, int]) -> None:
        """
        Sets the inner rectangle (params mask of the packet).

        Args:
            mask (tuple): Center(x, y), (width, height), angle of rotation.
        """

        if not isinstance(mask, np.ndarray):
            print(
                f"[WARN]: Tried to crop packet mask of type {type(mask)} (Not a np.ndarray)"
            )
            return

        # Update the mask
        if self.mask is None:
            self.mask = mask
        else:
            if mask.shape != self.mask.shape:
                print(f"[WARN]: Tried to average two uncompatible sizes")
                return
            self.mask = np.logical_and(mask, self.mask)

    def add_angle_to_average(self, angle: float) -> None:
        """
        Adds new angle value into the average angle of the packet.

        Args:
            angle (int | float): Angle of packet contours from horizontal line in the frame.
        """

        if not -90 <= angle <= 90:
            print(
                "[WARNING] Tried to add packet ANGLE with value {} (Should be between -90 and 90)".format(
                    angle
                )
            )
            return

        self.avg_angle_deg = angle

        # Update average
        # if self.avg_angle_deg is None:
        #     self.avg_angle_deg = angle
        # else:
        #     self.avg_angle_deg = (self.num_avg_angles * self.avg_angle_deg + angle) / (
        #         self.num_avg_angles + 1
        #     )

        # self.num_avg_angles += 1

    def add_depth_crop_to_average(self, depth_crop: np.ndarray) -> None:
        """
        Adds new depth crop into the average depth image of the packet.

        Args:
            depth_crop (np.ndarray): Frame containing depth values, has to have same size as previous depth frames
        """

        if self.avg_depth_crop is None:
            self.avg_depth_crop = depth_crop
        else:
            if not self.avg_depth_crop.shape == depth_crop.shape:
                print(
                    "[WARNING] Tried to average two depth maps with incompatible shape together: {} VS {}".format(
                        self.avg_depth_crop.shape, depth_crop.shape
                    )
                )
                return
            self.avg_depth_crop = (
                self.num_avg_depths * self.avg_depth_crop + depth_crop
            ) / (self.num_avg_depths + 1)

        self.num_avg_depths += 1

    def get_crop_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Crops packet with some small border around it (border given by self.crop_border_px) and returns the cropped array.
        If it is the first cropped depth map, packet width and height are used, otherwise the previous size is used.

        Args:
            frame (np.ndarray): Image frame with the packet values.

        Returns:
            np.ndarray: Numpy array containing the cropped packet.
        """

        if self.num_avg_depths == 0:
            crop = frame[
                int(
                    self.centroid_px.y
                    - self.bounding_height_px // 2
                    - self.crop_border_px
                ) : int(
                    self.centroid_px.y
                    + self.bounding_height_px // 2
                    + self.crop_border_px
                ),
                int(
                    self.centroid_px.x
                    - self.bounding_width_px // 2
                    - self.crop_border_px
                ) : int(
                    self.centroid_px.x
                    + self.bounding_width_px // 2
                    + self.crop_border_px
                ),
            ]
        else:
            crop = frame[
                int(self.centroid_px.y - self.avg_depth_crop.shape[0] // 2) : int(
                    self.centroid_px.y + self.avg_depth_crop.shape[0] // 2
                ),
                int(self.centroid_px.x - self.avg_depth_crop.shape[1] // 2) : int(
                    self.centroid_px.x + self.avg_depth_crop.shape[1] // 2
                ),
            ]

        return crop

    def get_centroid_in_px(self) -> tuple[int, int]:
        return self.centroid_px

    def get_centroid_in_mm(self) -> tuple[float, float]:
        # Transform centroid from pixels to centimeters using a homography matrix
        centroid_cm = np.matmul(
            self.homography_matrix,
            np.array([self.centroid_px.x, self.centroid_px.y, 1]),
        )
        # Transform centroid from centimeters to millimeters
        centroid_mm = self.PointTuple(centroid_cm[0] * CM2MM, centroid_cm[1] * CM2MM)
        return centroid_mm

    def get_centroid_from_encoder_in_px(
        self, encoder_position: float
    ) -> tuple[int, int]:
        # k is a constant for translating
        # from distance in mm computed from encoder data
        # into frame pixels for a specific resolution
        # k = 0.8299  # 640 x 480
        # k = 1.2365  # 1280 x 720
        k = 1.8672  # 1440 x 1080
        # k = 1.2365  # 1080 x 720

        centroid_encoder_px = self.PointTuple(
            int(
                k * (encoder_position - self.encoder_base_position)
                + self.centroid_base_px.x
            ),
            self.centroid_px.y,
        )

        return centroid_encoder_px

    def get_centroid_from_encoder_in_mm(
        self, encoder_position: float
    ) -> tuple[float, float]:
        centroid_robot_frame = np.matmul(
            self.homography_matrix,
            np.array([self.centroid_px.x, self.centroid_px.y, 1]),
        )

        packet_x = centroid_robot_frame[0] * CM2MM
        packet_y = centroid_robot_frame[1] * CM2MM
        return packet_x, packet_y

    def get_angle(self) -> float:
        return self.avg_angle_deg

    def get_width_in_px(self) -> int:
        return self.bounding_width_px

    def get_height_in_px(self) -> int:
        return self.bounding_height_px

    def get_width_in_mm(self) -> float:
        bounding_width_mm = (
            self.bounding_width_px
            * sqrt(
                self.homography_matrix[0, 0] ** 2 + self.homography_matrix[1, 0] ** 2
            )
            * CM2MM
        )
        return bounding_width_mm

    def get_height_in_mm(self) -> float:
        bounding_height_mm = (
            self.bounding_height_px
            * sqrt(
                self.homography_matrix[0, 1] ** 2 + self.homography_matrix[1, 1] ** 2
            )
            * CM2MM
        )
        return bounding_height_mm
