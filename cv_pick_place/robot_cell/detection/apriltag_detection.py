import json

import cv2
import numpy as np


class ProcessingApriltag:
    """
    Class for finding April Tags in image and calculating a homography matrix
    which transforms coordinates in pixels to coordinates defined by detected April Tags.
    """

    def __init__(self):
        """
        ProcessingApriltag object constructor.
        """

        self.image_points = {}
        self.world_points = {}
        self.world_points_detect = []
        self.image_points_detect = []
        self.homography = None
        self.tag_corner_list = None
        self.tag_id_list = None

    def load_world_points(self, file_path: str):
        """
        Loads conveyor world points from a json file.

        Args:
            file_path (str): Path to a json file containing coordinates.
        """

        with open(file_path, "r") as f:
            self.world_points = json.load(f)

    def compute_homog(self) -> np.ndarray:
        """
        Computes homography matrix using image and conveyor world points.

        Returns:
            np.ndarray: Homography matrix as numpy array.
        """

        for tag_id in self.image_points:
            if tag_id in self.world_points:
                self.world_points_detect.append(self.world_points[tag_id])
                self.image_points_detect.append(self.image_points[tag_id])

        # Only update homography matrix if enough points were detected
        is_enough_points_detect = len(self.image_points_detect) >= 4

        if is_enough_points_detect:
            self.homography, _ = cv2.findHomography(
                np.array(self.image_points_detect), np.array(self.world_points_detect)
            )
        else:
            print(
                "[WARNING]: Less than 4 AprilTags found in frame, new homography matrix was not computed"
            )

        return self.homography

    def detect_tags(self, color_frame: np.ndarray):
        """
        Detects april tags in the input image.

        Args:
            color_image (np.ndarray): Image where apriltags are to be detected.
        """

        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            gray_frame,
            cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11),
            parameters=cv2.aruco.DetectorParameters_create(),
        )

        # If nothing was detected, return
        if len(corners) == 0 or ids is None:
            return

        self.tag_corner_list = corners
        self.tag_id_list = ids.flatten()

        for (tag_corners, tag_id) in zip(self.tag_corner_list, self.tag_id_list):
            # Get (x, y) corners of the tag
            corners = tag_corners.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners

            top_left = (int(top_left[0]), int(top_left[1]))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

            # Compute centroid
            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)

            # Store detected points for homography computation
            self.image_points[str(int(tag_id))] = [cX, cY]

    def draw_tags(self, image_frame: np.ndarray) -> np.ndarray:
        """
        Draws detected april tags into image frame.

        Args:
            image_frame (np.ndarray): Image where apriltags are to be drawn.

        Returns:
            np.ndarray: Image with drawn april tags.
        """

        if not isinstance(image_frame, np.ndarray):
            print(
                "[WARNING] Tried to draw AprilTags into something which is not numpy.ndarray image frame"
            )
            return image_frame

        if self.tag_corner_list is None or self.tag_id_list is None:
            return image_frame

        cv2.polylines(image_frame, np.int0(self.tag_corner_list), True, (0, 255, 0), 2)

        for tag_id in self.tag_id_list:
            text = str(int(tag_id))
            cv2.putText(
                image_frame,
                text,
                (
                    self.image_points[str(int(tag_id))][0] + 30,
                    self.image_points[str(int(tag_id))][1],
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return image_frame
