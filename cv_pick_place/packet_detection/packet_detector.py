import random
import os
import numpy as np
import cv2
import matplotlib as mpl
import scipy.signal
from scipy import ndimage
from scipy.spatial import distance as dist
from collections import OrderedDict
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from cvzone.HandTrackingModule import HandDetector
from packet_object.packet import Packet

class PacketDetector:
    def __init__(self, paths, files, checkpt):
        """
        PacketDetector object constructor.
    
        Parameters:
        paths (dict): Dictionary with annotation and checkpoint paths.
        files (dict): Dictionary with pipeline and config paths. 
        checkpt (str): Name of training checkpoint to be restored. 

        """
        self.paths = paths
        self.files = files
        self.checkpt = checkpt
        self.world_centroid = None
        self.category_index = label_map_util.create_category_index_from_labelmap(
            self.files['LABELMAP'])
        configs = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=configs['model'], 
                                                    is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model= self.detection_model)
        ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], 
                                    self.checkpt)).expect_partial()

    @tf.function
    def detect_fn(self, image):
        """
        Neural net detection function.
    
        Parameters:
        image (tf.Tensor): Input image where objects are to be detected.

        Returns:
        dict: dictionary with detections.
        
        """
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def detect_corners(self, img):
        """
        Corner detection algorithm.
    
        Parameters:
        image (numpy.ndarray): Input image where corners are to be detected.

        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img,(x, y),4 , (255,0,0),-1)
        cv2.imshow("corners",img)

    def find_packet_contours(self, img, ymin, ymax, xmin, xmax, centroid):
        """
        Finds packet coutours.
    
        Parameters:
        image (numpy.ndarray): Input image where coutours are to be detected.
        ymin (int): Lower Y coordinate of bounding box.
        ymax (int): Upper Y coordinate of bounding box.
        xmin (int): Lower X coordinate of bounding box.
        xmax (int): Upper X coordinate of bounding box.
        centroid (tuple): Centroid to be updated.

        Returns:
        tuple: Points of the contour box, angle of rotation and updated centroid.
        
        """
        box = np.int64(np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]))
        angle = 0
        crop = img[int(ymin):int(ymax),int(xmin):int(xmax),:]
        
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
                cx , cy = x+xmin, y+ymin
                centroid = (int(cx), int(cy))
                box = cv2.boxPoints(((cx,cy),(w, h), angle))
                box = np.int0(box)
        return box, angle, centroid

    def compute_mask(self, img, box_mask, box_array):
        """
        Compute and packet mask and draw contours.
    
        Parameters:
        img (numpy.ndarray): Input image where coutours are to be detected.
        box_mask (numpy.ndarray): Output mask.
        box_array (list): list of contour box points.

        Returns:
        tuple: output image with contours, computed output mask.
        
        """
        is_box_empty = len(box_array) == 0
        if is_box_empty:
            return img, box_mask
        else:
            cv2.fillPoly(box_mask, box_array,(255, 255, 255))
            box_mask = cv2.GaussianBlur(box_mask, (5, 5), 0)
            cv2.polylines(img, box_array, True, (255, 0, 0), 3)
            return img, box_mask

    def deep_detector(self, color_frame, depth_frame, homography, bnd_box = True, segment = False):
        """
        Main packet detector function with homography transformation.
    
        Parameters:
        color_frame (numpy.ndarray): Input image where packets are to be detected.
        depth_frame (numpy.ndarray): Depth frame.
        homography (numpy.ndarray): homography matrix.
        bnd_box (bool): Bool to enable or disable bounding box visualization.

        Returns:
        tuple: Image with detected packets, segmented packets, detections.
        
        """
        box_array = []
        detected = []
        box_mask = np.zeros_like(color_frame)
        image_np = np.array(color_frame)
        height, width, depth = image_np.shape[0],image_np.shape[1],image_np.shape[2]
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        img_np_detect = image_np.copy()

        boxes = detections['detection_boxes']
        # get all boxes from an array
        # max_boxes_to_draw = boxes.shape[0]
        max_boxes_to_draw = 1
        # get scores to get a threshold
        scores = detections['detection_scores']
        # set as a default but free to adjust it to your needs
        min_score_thresh=.8
        # iterate over all objects found
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                # print ("This box is gonna get used", boxes[i], 
                        # detections['detection_classes'][i])
                ymin, xmin = boxes[i][0]*height, boxes[i][1]*width
                ymax, xmax = boxes[i][2]*height, boxes[i][3]*width
                cx,cy = (xmax+xmin)/2,(ymax+ymin)/2
                centroid = (int(cx),int(cy))
                box, angle, centroid = self.find_packet_contours(color_frame, 
                                                                    ymin, 
                                                                    ymax, 
                                                                    xmin, 
                                                                    xmax, 
                                                                    centroid)
                box_array.append(box)

                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0),5)
                distance = depth_frame[centroid[1], centroid[0]]
                cv2.putText(img_np_detect, "{} deg".format(round(angle, 1)), 
                            (centroid[0], centroid[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img_np_detect, "{}mm".format(distance), 
                            (centroid[0], centroid[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if homography is not None:
                    new_centroid = np.append(centroid,1)
                    self.world_centroid = homography.dot(new_centroid)
                    self.world_centroid = self.world_centroid[0], self.world_centroid[1]
                    cv2.putText(img_np_detect, 
                                str(round(self.world_centroid[0],2)) +','+ 
                                str(round(self.world_centroid[1],2)), centroid, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                detected.append([box, 
                                centroid, 
                                self.world_centroid, 
                                angle, 
                                detections['detection_classes'][i]])
        img_np_detect, box_mask = self.compute_mask(img_np_detect,box_mask, box_array)
                     
        if bnd_box:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                        img_np_detect,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        self.category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.8,
                        agnostic_mode=False, 
                        line_thickness=1)
        if segment:
            img_segmented = np.bitwise_and(color_frame,box_mask)
            # img_segmented = box_mask
            return img_np_detect, detected, img_segmented
        else:
            return img_np_detect, detected

    def deep_detector_v2(self, color_frame, depth_frame, bnd_box = True, segment = False):
        """
        Main packet detector function.
    
        Parameters:
        color_frame (numpy.ndarray): Input image where packets are to be detected.
        depth_frame (numpy.ndarray): Depth frame.
        bnd_box (bool): Bool to enable or disable bounding box visualization.

        Returns:
        tuple: Image with detected packets, segmented packets, detections.
        
        """
        box_array = []
        detected = []
        box_mask = np.zeros_like(color_frame)
        image_np = np.array(color_frame)
        height, width, depth = image_np.shape[0],image_np.shape[1],image_np.shape[2]
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        img_np_detect = image_np.copy()
        boxes = detections['detection_boxes']
        max_boxes_to_draw = 1
        # get scores to get a threshold
        scores = detections['detection_scores']
        # set as a default but free to adjust it to your needs
        min_score_thresh=.7
        # iterate over all objects found
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                # print ("This box is gonna get used", boxes[i], 
                        # detections['detection_classes'][i])
                ymin, xmin = boxes[i][0]*height, boxes[i][1]*width
                ymax, xmax = boxes[i][2]*height, boxes[i][3]*width
                cx,cy = (xmax+xmin)/2,(ymax+ymin)/2
                centroid = (int(cx),int(cy))
                box, angle, centroid = self.find_packet_contours(color_frame, 
                                                                    ymin, 
                                                                    ymax, 
                                                                    xmin, 
                                                                    xmax, 
                                                                    centroid)
                box_array.append(box)

                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0),5)
                distance = depth_frame[centroid[1], centroid[0]]
                cv2.putText(img_np_detect, "{} deg".format(round(angle, 1)), 
                            (centroid[0], centroid[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img_np_detect, "{}mm".format(distance), 
                            (centroid[0], centroid[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                detected.append([box, 
                                centroid, 
                                angle, 
                                detections['detection_classes'][i]])
        img_np_detect, box_mask = self.compute_mask(img_np_detect,box_mask, box_array)
                     
        if bnd_box:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                        img_np_detect,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        self.category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.7,
                        agnostic_mode=False, 
                        line_thickness=1)
        if segment:
            img_segmented = np.bitwise_and(color_frame,box_mask)
            # img_segmented = box_mask
            return img_np_detect, detected, img_segmented
        else:
            return img_np_detect, detected

    def deep_pack_obj_detector(self, color_frame, depth_frame, bnd_box = True, segment = False):
        """
        Main packet detector function.
    
        Parameters:
        color_frame (numpy.ndarray): Input image where packets are to be detected.
        depth_frame (numpy.ndarray): Depth frame.
        bnd_box (bool): Bool to enable or disable bounding box visualization.

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
        height, width, depth = image_np.shape[0],image_np.shape[1],image_np.shape[2]
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        img_np_detect = image_np.copy()
        boxes = detections['detection_boxes']
        max_boxes_to_draw = 1
        # get scores to get a threshold
        scores = detections['detection_scores']
        # set as a default but free to adjust it to your needs
        min_score_thresh=.7
        # iterate over all objects found
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                # print ("This box is gonna get used", boxes[i], 
                        # detections['detection_classes'][i])
                ymin, xmin = boxes[i][0]*height, boxes[i][1]*width
                ymax, xmax = boxes[i][2]*height, boxes[i][3]*width
                cx,cy = (xmax+xmin)/2,(ymax+ymin)/2
                centroid = (int(cx),int(cy))
                w = float((xmax - xmin)) / width
                h = float((ymax - ymin)) / height
                box, angle, centroid = self.find_packet_contours(color_frame, 
                                                                    ymin, 
                                                                    ymax, 
                                                                    xmin, 
                                                                    xmax, 
                                                                    centroid)
                box_array.append(box)

                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0),5)
                distance = depth_frame[centroid[1], centroid[0]]
                cv2.putText(img_np_detect, "{} deg".format(round(angle, 1)), 
                            (centroid[0], centroid[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img_np_detect, "{}mm".format(distance), 
                            (centroid[0], centroid[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                packet = Packet(box = box, 
                            pack_type = detections['detection_classes'][i],
                            centroid = centroid, 
                            angle = angle,
                            width = w, height= h)
                if centroid[0] - w/2 > guard  and centroid[0] + w/2 < (width - guard ):
                    detected.append(packet)
        img_np_detect, box_mask = self.compute_mask(img_np_detect,box_mask, box_array)
                     
        if bnd_box:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                        img_np_detect,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        self.category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.7,
                        agnostic_mode=False, 
                        line_thickness=1)
        if segment:
            img_segmented = np.bitwise_and(color_frame,box_mask)
            # img_segmented = box_mask
            return img_np_detect, detected, img_segmented
        else:
            return img_np_detect, detected