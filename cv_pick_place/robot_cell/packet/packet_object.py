from math import sqrt
import numpy as np
from collections import namedtuple


class Packet:
    """
    Class wrapping relevant information about packets together.
    """

    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        box: np.ndarray = np.empty(()),
        crop_border_px: int = 50,
    ):
        """
        Packet object constructor.

        Args:
            box (np.ndarray): Packet bounding box.
            centroid (tuple): Centroid of packet.
            centroid_depth (int): Depth of centroid from depth frame.
            angle (float): Angle of bounding box.
            width (int): Width of bounding box.
            height (int): Height of bounding box.
            ymin (int):
            ymax (int):
            xmin (int):
            xmax (int):
            encoder_position (float): Position of encoder.
            crop_border_px (int): Border around the packet that is included in the depth crop.
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
        # Homography matrix used to transform packet data in pixels to millimeters
        self.homography = None

        # X Y centroid value in frame pixels
        # X increases from left to right
        # Y increases from top to bottom
        self.centroid_px = None

        # Width of horizontally aligned bounding box in pixels
        self.width_bnd_px = None
        # Width of horizontally aligned bounding box in milimeters
        self.width_bnd_mm = None
        # Height of horizontally aligned bounding box in pixels
        self.height_bnd_px = None
        # Height of horizontally aligned bounding box in milimeters
        self.height_bnd_mm = None

        # Binary mask for the depth detection
        self.mask = None

        # Ammount of extra pixels around each depth crop
        self.crop_border_px = crop_border_px
        # Number of averaged depth crops
        self.num_avg_depths = 0
        # Numpy array with average depth value for the packet
        self.avg_depth_crop = None

        # Number of averaged angles
        self.num_avg_angles = 0
        # Number with average angle of the packet
        # The angle gives the rotation of packet contours from horizontal line in the frame
        self.avg_angle_deg = None

        # First detected position of packet centroid in pixels
        self.centroid_initial_px = None
        # First detected position of encoder in pixels
        self.encoder_initial_position = None
        # Number of frames item has been tracked
        self.track_frame = 0

        self.in_pick_list = False

        # Number of frames the packet has disappeared for
        self.disappeared = 0

        # OBSOLETE PACKET PARAMETERS
        ############################

        # Width and height of packet bounding box
        self.width = width
        self.height = height

        self.box = box

    def set_id(self, packet_id: int) -> None:
        """
        Sets packet ID.

        Args:
            packet_id (int): Unique integer greater then or equal to 0.
        """

        if not isinstance(packet_id, int):
            print(
                f"[WARNING] Tried to set packet ID to {packet_id} (Should be integer)"
            )
            return

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

        # Check input validity
        if not isinstance(packet_type, int):
            print(
                f"[WARNING] Tried to set packet TYPE to {packet_type} (Should be integer)"
            )
            return

        # Set parameter
        self.type = packet_type

    def set_centroid(self, x: int, y: int) -> None:
        """
        Sets packet centroid.

        Args:
            x (int): X coordinate of centroid in pixels.
            y (int): Y coordinate of centroid in pixels.
        """

        # Check input validity
        if not isinstance(x, int) or not isinstance(y, int):
            print(
                "[WARNING] Tried to set packet CENTROID to ({}, {}) (Should be integers)".format(
                    x, y
                )
            )
            return

        # Set parameter
        self.centroid_px = self.PointTuple(x, y)

    def set_homography(self, homography: np.ndarray) -> None:
        """
        Sets packet homography.

        Args:
            homography (int): 3x3 homography matrix converting from pixels to millimeters.
        """
        if not isinstance(homography, np.ndarray):
            print(
                f"[WARNING] Tried set HOMOGRAPHY to \n{homography} \n(Not a np.ndarray)"
            )
            return

        self.homography = homography

    def set_base_encoder_position(self, encoder_position: float) -> None:
        """
        Sets packet initial encoder position and associated centroid position.

        Args:
            encoder_pos (float): Position of encoder.
        """
        if not isinstance(encoder_position, float):
            print(
                f"[WARNING] Tried to set packet ENCODER_POSITION to {encoder_position} (Should be float)"
            )
            return

        self.centroid_initial_px = self.centroid_px
        self.encoder_initial_position = encoder_position

    def set_bounding_size(
        self, width: int, height: int, homography: np.ndarray = None
    ) -> None:
        """
        Sets width and height of packet bounding box.

        Args:
            width (int): Width of the packet in pixels.
            height (int): Height of the packet in pixels.
            homography (np.ndarray): Homography matrix converting from pixels to centimeters.
        """

        # Check input validity
        if not isinstance(width, int) or not isinstance(height, int):
            print(
                "[WARNING] Tried to set packet WIDTH and HEIGHT to ({}, {}) (Should be integers)".format(
                    width, height
                )
            )
            return

        if width <= 0 or height <= 0:
            print(
                "[WARNING] Tried to set packet WIDTH and HEIGHT to ({}, {}) (Should be greater than 0)".format(
                    width, height
                )
            )
            return

        # Set parameters
        self.width_bnd_px = width
        self.height_bnd_px = height

        # Compute parameter in world coordinates
        if homography is not None:
            if not isinstance(homography, np.ndarray):
                print(
                    "[WARNING] Tried to compute packet WIDTH and HEIGHT with homography \n{} \n(Not a np.ndarray)".format(
                        homography
                    )
                )
                return

            self.width_bnd_mm = (
                width * sqrt(homography[0, 0] ** 2 + homography[1, 0] ** 2) * 10
            )
            self.height_bnd_mm = (
                height * sqrt(homography[0, 1] ** 2 + homography[1, 1] ** 2) * 10
            )
        else:
            self.width_bnd_mm = None
            self.height_bnd_mm = None

        # OBSOLETE PARAMS
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

        # Check input validity
        if not isinstance(angle, int) and not isinstance(angle, float):
            print(
                "[WARNING] Tried to add packet ANGLE with value {} (Should be float or integer)".format(
                    angle
                )
            )
            return

        if not -90 <= angle <= 90:
            print(
                "[WARNING] Tried to add packet ANGLE with value {} (Should be between -90 and 90)".format(
                    angle
                )
            )
            return

        # Update average
        if self.avg_angle_deg is None:
            self.avg_angle_deg = angle
        else:
            self.avg_angle_deg = (self.num_avg_angles * self.avg_angle_deg + angle) / (
                self.num_avg_angles + 1
            )

        self.num_avg_angles += 1

    def add_depth_crop_to_average(self, depth_crop: np.ndarray) -> None:
        """
        Adds new depth crop into the average depth image of the packet.

        Args:
            depth_crop (np.ndarray): Frame containing depth values, has to have same size as previous depth frames
        """

        # Check input validity
        if not isinstance(depth_crop, np.ndarray):
            print(
                "[WARNING] Tried to add packet CROP of type {} (Not a np.ndarray)".format(
                    type(depth_crop)
                )
            )
            return

        # Update average
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

        # Check input validity
        if not isinstance(frame, np.ndarray):
            print(
                "[WARNING] Tried to crop frame of type {} (Not a np.ndarray)".format(
                    type(frame)
                )
            )
            return None

        # Compute crop
        if self.num_avg_depths == 0:
            crop = frame[
                int(self.centroid_px.y - self.height // 2 - self.crop_border_px) : int(
                    self.centroid_px.y + self.height // 2 + self.crop_border_px
                ),
                int(self.centroid_px.x - self.width // 2 - self.crop_border_px) : int(
                    self.centroid_px.x + self.width // 2 + self.crop_border_px
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
        transformed_centroid = np.matmul(
            self.homography, np.array([self.centroid_px.x, self.centroid_px.y, 1])
        )
        centroid_mm = self.PointTuple(transformed_centroid[0] * 10, transformed_centroid[1] * 10)
        return centroid_mm

    def getCentroidFromEncoder(self, encoder_position: float) -> tuple[float, float]:
        """
        Computes actual centroid of packet from the encoder data.

        Args:
            encoder_position (float): current encoder position.

        Returns:
            tuple[float, float]: Updated x, y packet centroid.

        """
        # k = 0.8299  # 640 x 480
        # k = 1.2365  # 1280 x 720
        k = 1.8672  # 1440 x 1080
        # k = 1.2365  # 1080 x 720
        return (
            int(
                k * (encoder_position - self.encoder_initial_position)
                + self.centroid_initial_px.x
            ),
            self.centroid_px.y,
        )

    # OBSOLETE
    def getCentroidInWorldFrame(self, homography: np.ndarray) -> float:
        """
        Converts centroid from image coordinates to real world coordinates.

        Args:
            homography (np.ndarray): homography matrix.

        Returns:
            float: Updated x, y packet centroid in world coordinates.
        """

        centroid_robot_frame = np.matmul(
            homography, np.array([self.centroid_px.x, self.centroid_px.y, 1])
        )

        packet_x = centroid_robot_frame[0] * 10
        packet_y = centroid_robot_frame[1] * 10
        return packet_x, packet_y
