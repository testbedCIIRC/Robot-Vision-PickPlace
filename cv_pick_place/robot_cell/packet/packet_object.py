from math import sqrt
import numpy as np
from collections import namedtuple
import cv2

class Packet:
    """
    Class wrapping relevant information about packets together.
    """

    def __init__(
        self,
        box: np.ndarray = np.empty(()),
        pack_type: int = None,
        centroid: tuple[int, int] = (0, 0),
        centroid_depth: int = 0,
        angle: float = 0,
        width: int = 0,
        height: int = 0,
        ymin: int = 0,
        ymax: int = 0,
        xmin: int = 0,
        xmax: int = 0,
        encoder_position: float = 0,
        crop_border_px: int = 50,
    ):
        """
        Packet object constructor.

        Args:
            box (np.ndarray): Packet bounding box.
            pack_type (bool): Class of detected packet.
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

        # X Y centroid value in frame pixels
        # X increases from left to right
        # Y increases from top to bottom
        self.centroid_px = None
        # X Y centroid value in milimeters
        # Depends on detected apriltags and computed homography
        self.centroid_mm = None

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

        # Mask used for optimal point from depth crop
        self.img_maks = None
        self.num_proccessed_mask = 0

        # OBSOLETE PACKET PARAMETERS
        ############################
        # Tuple of 2 numbers describing center of the packet
        self.centroid = centroid

        # List of angles for averaging
        self.angles = [angle]
        self.angle = angle

        # Width and height of packet bounding box
        self.width = width
        self.height = height

        self.yminbbx = ymin
        self.ymaxbbx = ymax
        self.xminbbx = xmin
        self.xmaxbbx = xmax

        # Numpy array of cropped depth maps
        self.depth_maps = np.empty(())
        self.color_frames = np.empty(())

        # Number of frames the packet has disappeared for
        self.disappeared = 0

        self.time_of_disappearance = None

        self.box = box

        self.pack_type = pack_type

        # Encoder data
        self.starting_encoder_position = encoder_position

        # Used for offsetting centroid position calculated using encoder
        self.first_centroid_position = centroid

    def set_id(self, id: int) -> None:
        """
        Sets packet ID.

        Args:
            id (int): Unique integer greater then or equal to 0.
        """

        if not isinstance(id, int):
            print(
                "[WARNING] Tried to set packet ID to {} (Should be integer)".format(id)
            )
            return

        if id < 0:
            print(
                "[WARNING] Tried to set packet ID to {} (Should be greater than or equal to 0)".format(
                    id
                )
            )
            return

        self.id = id

    def set_type(self, type: int) -> None:
        """
        Sets packet type.

        Args:
            type (int): Integer representing type of the packet.
        """

        # Check input validity
        if not isinstance(type, int):
            print(
                "[WARNING] Tried to set packet TYPE to {} (Should be integer)".format(
                    type
                )
            )
            return

        # Set parameter
        self.type = type

        # OBSOLETE PARAM
        self.pack_type = type

    def set_centroid(
        self, x: int, y: int, homography: np.ndarray = None, encoder_pos: int = None
    ) -> None:
        """
        Sets packet centroid.

        Args:
            x (int): X coordinate of centroid in pixels.
            x (int): Y coordinate of centroid in pixels.
            homography (np.ndarray): Homography matrix converting from pixels to centimeters.
            encoder_pos (float): Position of encoder.
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

        # Set initial values if not set
        if self.centroid_initial_px is None and encoder_pos is not None:
            self.centroid_initial_px = self.centroid_px
            self.encoder_initial_position = encoder_pos

        # Compute parameter in world coordinates
        if homography is not None:
            if not isinstance(homography, np.ndarray):
                print(
                    "[WARNING] Tried to compute packet CENTROID with homography \n{} \n(Not a np.ndarray)".format(
                        homography
                    )
                )
                return

            transformed_centroid = np.matmul(homography, np.array([x, y, 1]))
            self.centroid_mm = self.PointTuple(
                transformed_centroid[0] * 10, transformed_centroid[1] * 10
            )
        else:
            self.centroid_mm = self.PointTuple(None, None)

        # OBSOLETE PARAM
        self.centroid = (x, y)

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

    def set_img_mask(self, img_mask: np.ndarray) -> None:
        """
        Sets binary mask of the image.

        Args:
            img_mask (np.ndarray): Image binary mask of the whole image.

        """
        self.full_img_mask = (img_mask > 0) * 1.0

    def add_mask_to_average(self, mask: np.ndarray) -> None:
        """
        Add new binary mask to average. If no mask have been saved, saves
        new mask as original

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
            self.num_proccessed_mask = 1
        else:
            if mask.shape != self.mask.shape:
                print(f"[WARN]: Tried to average two uncompatible sizes {mask.shape} and {self.mask.shape}")
                return
            # Some kind of averaging
            self.mask = self.num_proccessed_mask * self.mask + mask
            self.num_proccessed_mask += 1
            self.mask  = self.mask / self.num_proccessed_mask
            # This was the old way
            # self.mask = np.logical_and(mask, self.mask)
            # print(np.unique(self.mask, return_counts=True))

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

    def getCentroidFromEncoder(self, encoder_position: float) -> tuple[float, float]:
        """
        Computes actual centroid of packet from the encoder data.

        Args:
            encoder_position (float): current encoder position.

        Returns:
            tuple[float, float]: Updated x, y packet centroid.

        """
        # k = 1.8672  # 1440 x 1080 and 1920 x 1080
        k = 0.9323 # 960 x 540
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
            homography, np.array([self.centroid[0], self.centroid[1], 1])
        )

        packet_x = centroid_robot_frame[0] * 10
        packet_y = centroid_robot_frame[1] * 10
        return packet_x, packet_y
