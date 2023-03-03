import os

import numpy as np
import cv2

from robot_cell.packet.packet_object import Packet


class PacketDetector:
    """
    Class for detecting packets using neural network.
    """

    def __init__(
        self,
        annotation_path: str,
        checkpoint_path: str,
        pipeline_config_file: str,
        labelmap_file: str,
        checkpt: str,
        max_detect: int = 1,
        detect_thres: float = 0.7,
    ):
        """
        PacketDetector object constructor.

        Args:
            annotation_path (str): Path to detector annotations.
            checkpoint_path (str): Path to detector checkpoints.
            pipeline_config_file (str): Path to detector configuration file.
            labelmap_file (str): Path to detector labelmap file.
            checkpt (str): Name of training checkpoint to be restored.
            max_detect (int): Maximal ammount of concurrent detections in an image.
            detect_thres (float): Minimal confidence for detected object to be labeled as a packet.
        """

        # Import the tf detection dependencies
        self.import_detection_libs()
        # Decorate the detection function with the tf.function decorator
        tf_func_decorator = self.tf.function()
        self.detect_fn = tf_func_decorator(self.detect_fn)
        self.checkpt = checkpt
        self.max_detect = max_detect
        self.detect_thres = detect_thres
        self.world_centroid = None
        self.category_index = self.label_map_util.create_category_index_from_labelmap(
            labelmap_file
        )
        configs = self.config_util.get_configs_from_pipeline_file(pipeline_config_file)
        self.detection_model = self.model_builder.build(
            model_config=configs["model"], is_training=False
        )
        # Restore checkpoint
        ckpt = self.tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(checkpoint_path, self.checkpt)).expect_partial()

    def import_detection_libs(self):
        """
        Imports the tensorflow detection dependencies.
        """

        import tensorflow as tf
        from object_detection.utils import config_util
        from object_detection.utils import label_map_util
        from object_detection.builders import model_builder
        from object_detection.utils import visualization_utils as viz_utils

        self.tf = tf
        self.config_util = config_util
        self.label_map_util = label_map_util
        self.model_builder = model_builder
        self.viz_utils = viz_utils

    def detect_fn(self, image) -> dict:
        """
        Neural net detection function.

        Args:
            image (tf.Tensor): Input image where objects are to be detected.

        Returns:
            dict: Dictionary with detections.
        """

        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def detect_corners(self, img: np.ndarray):
        """
        Corner detection algorithm.

        Args:
            image (np.ndarray): Input image where corners are to be detected.
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        cv2.imshow("corners", img)

    def find_packet_contours(
        self,
        img: np.ndarray,
        ymin: int,
        ymax: int,
        xmin: int,
        xmax: int,
        centroid: tuple,
    ) -> tuple:
        """
        Finds packet coutours.

        Args:
            image (np.ndarray): Input image where coutours are to be detected.
            ymin (int): Lower Y coordinate of bounding box.
            ymax (int): Upper Y coordinate of bounding box.
            xmin (int): Lower X coordinate of bounding box.
            xmax (int): Upper X coordinate of bounding box.
            centroid (tuple): Centroid to be updated.

        Returns:
            tuple: Points of the contour box, angle of rotation and updated centroid.
        """

        box = np.int64(
            np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        )
        angle = 0
        crop = img[int(ymin) : int(ymax), int(xmin) : int(xmax), :]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, mask = cv2.threshold(gray, 60, 255, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # img = cv2.drawContours(img, contours, -1, (0,255,0), 3,lineType = cv2.LINE_AA)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 8500:
                # if area > 25000 and area < 110000:
                # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                cx, cy = x + xmin, y + ymin
                centroid = (int(cx), int(cy))
                box = cv2.boxPoints(((cx, cy), (w, h), angle))
                box = np.int0(box)
        return box, angle, centroid

    def compute_mask(
        self, img: np.ndarray, box_mask: np.ndarray, box_array: list
    ) -> tuple:
        """
        Compute and packet mask and draw contours.

        Args:
            img (np.ndarray): Input image where coutours are to be detected.
            box_mask (np.ndarray): Output mask.
            box_array (list): List of contour box points.

        Returns:
            tuple: Output image with contours, computed output mask.
        """

        is_box_empty = len(box_array) == 0
        if is_box_empty:
            return img, box_mask
        else:
            cv2.fillPoly(box_mask, box_array, (255, 255, 255))
            box_mask = cv2.GaussianBlur(box_mask, (5, 5), 0)
            cv2.polylines(img, box_array, True, (255, 0, 0), 3)
            return img, box_mask

    def deep_detector(
        self,
        color_frame: np.ndarray,
        depth_frame: np.ndarray,
        homography: np.ndarray,
        bnd_box: bool = True,
        segment: bool = False,
    ):
        """
        Main packet detector function with homography transformation.

        Args:
            color_frame (np.ndarray): Input image where packets are to be detected.
            depth_frame (np.ndarray): Depth frame.
            homography (np.ndarray): homography matrix.
            bnd_box (bool): Bool to enable or disable bounding box visualization.
            segment (bool): Bool to enable or disable segmentation mask visualization.

        Returns:
            tuple: Image with detected packets, segmented packets, detections.
        """

        box_array = []
        detected = []
        box_mask = np.zeros_like(color_frame)
        image_np = np.array(color_frame)
        height, width, depth = image_np.shape[0], image_np.shape[1], image_np.shape[2]
        input_tensor = self.tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=self.tf.float32
        )
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections
        # detection_classes should be ints
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )
        label_id_offset = 1
        img_np_detect = image_np.copy()

        boxes = detections["detection_boxes"]
        # get all boxes from an array
        # max_boxes_to_draw = boxes.shape[0]
        max_boxes_to_draw = self.max_detect
        # get scores to get a threshold
        scores = detections["detection_scores"]
        # set as a default but free to adjust it to your needs
        min_score_thresh = self.detect_thres
        # iterate over all objects found
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):

            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                # print ("This box is gonna get used", boxes[i],
                # detections['detection_classes'][i])
                ymin, xmin = boxes[i][0] * height, boxes[i][1] * width
                ymax, xmax = boxes[i][2] * height, boxes[i][3] * width
                cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
                centroid = (int(cx), int(cy))
                box, angle, centroid = self.find_packet_contours(
                    color_frame, ymin, ymax, xmin, xmax, centroid
                )
                box_array.append(box)

                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0), 5)
                distance = depth_frame[centroid[1], centroid[0]]
                cv2.putText(
                    img_np_detect,
                    "{} deg".format(round(angle, 1)),
                    (centroid[0], centroid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    img_np_detect,
                    "{}mm".format(distance),
                    (centroid[0], centroid[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                if homography is not None:
                    new_centroid = np.append(centroid, 1)
                    self.world_centroid = homography.dot(new_centroid)
                    self.world_centroid = self.world_centroid[0], self.world_centroid[1]
                    cv2.putText(
                        img_np_detect,
                        str(round(self.world_centroid[0], 2))
                        + ","
                        + str(round(self.world_centroid[1], 2)),
                        centroid,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                detected.append(
                    [
                        box,
                        centroid,
                        self.world_centroid,
                        angle,
                        detections["detection_classes"][i],
                    ]
                )
        img_np_detect, box_mask = self.compute_mask(img_np_detect, box_mask, box_array)

        if bnd_box:
            self.viz_utils.visualize_boxes_and_labels_on_image_array(
                img_np_detect,
                detections["detection_boxes"],
                detections["detection_classes"] + label_id_offset,
                detections["detection_scores"],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=max_boxes_to_draw,
                min_score_thresh=min_score_thresh,
                agnostic_mode=False,
                line_thickness=1,
            )
        if segment:
            img_segmented = np.bitwise_and(color_frame, box_mask)
            # img_segmented = box_mask
            return img_np_detect, detected, img_segmented
        else:
            return img_np_detect, detected

    def deep_detector_v2(
        self,
        color_frame: np.ndarray,
        depth_frame: np.ndarray,
        bnd_box: bool = True,
        segment: bool = False,
    ) -> tuple:
        """
        Main packet detector function.

        Args:
            color_frame (np.ndarray): Input image where packets are to be detected.
            depth_frame (np.ndarray): Depth frame.
            bnd_box (bool): Bool to enable or disable bounding box visualization.
            segment (bool): Bool to enable or disable segmentation mask visualization.

        Returns:
            tuple: Image with detected packets, segmented packets, detections.
        """

        box_array = []
        detected = []
        box_mask = np.zeros_like(color_frame)
        image_np = np.array(color_frame)
        height, width, depth = image_np.shape[0], image_np.shape[1], image_np.shape[2]
        input_tensor = self.tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=self.tf.float32
        )
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )
        label_id_offset = 1
        img_np_detect = image_np.copy()
        boxes = detections["detection_boxes"]
        max_boxes_to_draw = self.max_detect
        # get scores to get a threshold
        scores = detections["detection_scores"]
        # set as a default but free to adjust it to your needs
        min_score_thresh = self.detect_thres
        # iterate over all objects found
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):

            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                # print ("This box is gonna get used", boxes[i],
                # detections['detection_classes'][i])
                ymin, xmin = boxes[i][0] * height, boxes[i][1] * width
                ymax, xmax = boxes[i][2] * height, boxes[i][3] * width
                cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
                centroid = (int(cx), int(cy))
                box, angle, centroid = self.find_packet_contours(
                    color_frame, ymin, ymax, xmin, xmax, centroid
                )
                box_array.append(box)

                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0), 5)
                distance = depth_frame[centroid[1], centroid[0]]
                cv2.putText(
                    img_np_detect,
                    "{} deg".format(round(angle, 1)),
                    (centroid[0], centroid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    img_np_detect,
                    "{}mm".format(distance),
                    (centroid[0], centroid[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                detected.append(
                    [box, centroid, angle, detections["detection_classes"][i]]
                )
        img_np_detect, box_mask = self.compute_mask(img_np_detect, box_mask, box_array)

        if bnd_box:
            self.viz_utils.visualize_boxes_and_labels_on_image_array(
                img_np_detect,
                detections["detection_boxes"],
                detections["detection_classes"] + label_id_offset,
                detections["detection_scores"],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=max_boxes_to_draw,
                min_score_thresh=min_score_thresh,
                agnostic_mode=False,
                line_thickness=1,
            )
        if segment:
            img_segmented = np.bitwise_and(color_frame, box_mask)
            # img_segmented = box_mask
            return img_np_detect, detected, img_segmented
        else:
            return img_np_detect, detected

    def deep_pack_obj_detector(
        self,
        color_frame: np.ndarray,
        depth_frame: np.ndarray,
        encoder_pos: float,
        bnd_box: bool = True,
        segment: bool = False,
        homography: np.ndarray = None,
        image_frame: np.ndarray = None,
    ) -> tuple:
        """
        Main packet detector function.

        Args:
            color_frame (np.ndarray): Input image where packets are to be detected.
            depth_frame (np.ndarray): Depth frame.
            encoder_pos (float): Current encoder position.
            bnd_box (bool): Bool to enable or disable bounding box visualization.
            segment (bool): Bool to enable or disable segmentation mask visualization.
            homography (np.ndarray): Homography matrix.
            image_frame (np.ndarray): Image frame into which information should be drawn.

        Returns:
            tuple: Image with detected packets, segmented packets, detections.
        """

        # Crop guard region
        # When packet depth is cropped, the resulting crop will
        # have 'guard' extra pixels on each side
        guard = 250
        box_array = []
        detected = []
        box_mask = np.zeros_like(color_frame)
        image_np = np.array(color_frame)
        height, width, depth = image_np.shape[0], image_np.shape[1], image_np.shape[2]
        input_tensor = self.tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=self.tf.float32
        )
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )
        label_id_offset = 1

        if image_frame is None or not image_frame.shape == color_frame.shape:
            img_np_detect = image_np.copy()
        else:
            img_np_detect = image_frame

        boxes = detections["detection_boxes"]
        max_boxes_to_draw = self.max_detect
        # get scores to get a threshold
        scores = detections["detection_scores"]
        # set as a default but free to adjust it to your needs
        min_score_thresh = self.detect_thres
        # iterate over all objects found
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):

            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                # print ("This box is gonna get used", boxes[i],
                # detections['detection_classes'][i])
                ymin, xmin = boxes[i][0] * height, boxes[i][1] * width
                ymax, xmax = boxes[i][2] * height, boxes[i][3] * width
                cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
                centroid = (int(cx), int(cy))
                w = float((xmax - xmin)) / width
                h = float((ymax - ymin)) / height
                box, angle, centroid = self.find_packet_contours(
                    color_frame, ymin, ymax, xmin, xmax, centroid
                )
                box_array.append(box)

                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0), 5)
                distance = depth_frame[centroid[1], centroid[0]]
                cv2.putText(
                    img_np_detect,
                    "{} deg".format(round(angle, 1)),
                    (centroid[0], centroid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    img_np_detect,
                    "{}mm".format(distance),
                    (centroid[0], centroid[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                packet = Packet()
                packet.set_type(int(detections["detection_classes"][i]))
                packet.set_centroid(centroid[0], centroid[1])
                packet.set_base_encoder_position(encoder_pos)
                packet.set_bounding_size(int(w * width), int(h * height))
                packet.add_angle_to_average(angle)
                if centroid[0] - w / 2 > guard and centroid[0] + w / 2 < (
                    width - guard
                ):
                    detected.append(packet)
        img_np_detect, box_mask = self.compute_mask(img_np_detect, box_mask, box_array)

        if bnd_box:
            self.viz_utils.visualize_boxes_and_labels_on_image_array(
                img_np_detect,
                detections["detection_boxes"],
                detections["detection_classes"] + label_id_offset,
                detections["detection_scores"],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=max_boxes_to_draw,
                min_score_thresh=min_score_thresh,
                agnostic_mode=False,
                line_thickness=1,
            )
        if segment:
            img_segmented = np.bitwise_and(color_frame, box_mask)
            # img_segmented = box_mask
            return img_np_detect, detected, img_segmented
        else:
            return img_np_detect, detected
