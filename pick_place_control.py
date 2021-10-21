from opcua import Client
import datetime
import time
import sys
import os
import json
import cv2 
from opcua import ua
import pyrealsense2
import numpy as np
from realsense_depth import *
import random
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
class processing_apriltag:
	
	def __init__(self,intrinsic,color_image,depth_frame):
		self.color_image = color_image
		self.intrinsic = intrinsic
		self.depth_frame = depth_frame
		self.radius = 20
		self.axis = 0
		self.packet1 = 24
		self.image_points = {}
		self.world_points = {}
		self.world_points_detect = []
		self.image_points_detect = []
		self.homography= None
	
	def load_original_points(self):
		f = open('points.json')
		# Dict of points in conveyor:
		self.world_points = json.load(f) 

	def compute_homog(self):
		for tag_id in self.image_points:
			if tag_id in self.world_points:
				self.world_points_detect.append(self.world_points[tag_id])
				self.image_points_detect.append(self.image_points[tag_id])
		is_enough_points_detect = len(self.image_points_detect)>= 4
		if is_enough_points_detect:
			self.homography,status = cv2.findHomography(np.array(self.image_points_detect), 
												np.array(self.world_points_detect))
			return self.homography
		else:
			print("[INFO]: Less than 4 corresponding points found")
			return self.homography

	def detect_tags(self):
		rect1 = []
		image = self.color_image.copy()
		self.load_original_points()
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		(corners, ids, rejected) = cv2.aruco.detectMarkers( image, cv2.aruco.Dictionary_get(
                                    cv2.aruco.DICT_APRILTAG_36h11),
                                    parameters=cv2.aruco.DetectorParameters_create())
		ids = ids.flatten()
		int_corners = np.int0(corners)
		cv2.polylines(image, int_corners, True, (0, 255, 0), 2)
        
		for (tag_corner, tag_id) in zip(corners, ids):
			# get (x,y) corners of tag
			corners = tag_corner.reshape((4, 2))
			(top_left, top_right, bottom_right, bottom_left) = corners
			top_right, bottom_right = (int(top_right[0]), int(top_right[1])),\
									  (int(bottom_right[0]), int(bottom_right[1]))
			bottom_left, top_left = (int(bottom_left[0]), int(bottom_left[1])),\
									(int(top_left[0]), int(top_left[1]))
			# compute centroid
			cX = int((top_left[0] + bottom_right[0]) / 2.0)
			cY = int((top_left[1] + bottom_right[1]) / 2.0)
			self.image_points[str(int(tag_id))] = [cX,cY]
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw ID on frame
			cv2.putText(image, str(tag_id),(top_left[0], top_left[1] - 15),
						cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
		return image

class CentroidTracker:
	def __init__(self, maxDisappeared=20):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# is box empty
		if len(rects) == 0:
			# loop overobjects and mark them as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
			#deregister if maximum number of consecutive frames where missing
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			# return early as there are no centroids or tracking info
			# to update
			return self.objects
		# array of input centroids at current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		# inputCentroids = centroid
		# # loop over the bounding box rectangles
		for i in range(0,len(rects)):
			# # use the bounding box coordinates to derive the centroid
			inputCentroids[i] = rects[i][1]
		# if not tracking any objects take input centroids, register them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		# else, while currently tracking objects try match the input centroids to existing centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()
			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue
				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)
			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
		# return the set of trackable objects
		return self.objects

class packet_detector:
    def __init__(self,paths,files,checkpt):
        self.paths = paths
        self.files = files
        self.checkpt = checkpt
        self.world_centroid = None
        self.category_index = label_map_util.create_category_index_from_labelmap(self.files['LABELMAP'])
        configs = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model= self.detection_model)
        ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], self.checkpt)).expect_partial()

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def detect_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img,(x, y),4 , (255,0,0),-1)
        cv2.imshow("corners",img)

    def find_packet_contours(self, img, ymin, ymax, xmin, xmax, centroid):
        box = np.int64(np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]))
        # box = None
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
        is_box_empty = len(box_array) == 0
        if is_box_empty:
            return img, box_mask
        else:
            cv2.fillPoly(box_mask, box_array,(255, 255, 255))
            box_mask = cv2.GaussianBlur(box_mask, (5, 5), 0)
            cv2.polylines(img, box_array, True, (255, 0, 0), 3)
            return img, box_mask

    def deep_detector(self, color_frame, depth_frame, homography, bnd_box = True):
        box_array = []
        rects = []
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
                # print ("This box is gonna get used", boxes[i], detections['detection_classes'][i])
                ymin, xmin = boxes[i][0]*height, boxes[i][1]*width
                ymax, xmax = boxes[i][2]*height, boxes[i][3]*width
                cx,cy = (xmax+xmin)/2,(ymax+ymin)/2
                centroid = (int(cx),int(cy))
                box, angle, centroid = self.find_packet_contours(color_frame, ymin, ymax, xmin, xmax, centroid)
                box_array.append(box)

                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0),5)
                distance = depth_frame[centroid[1], centroid[0]]
                cv2.putText(img_np_detect, "{} deg".format(round(angle, 1)), (centroid[0], centroid[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img_np_detect, "{}mm".format(distance), (centroid[0], centroid[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if homography is not None:
                    new_centroid = np.append(centroid,1)
                    self.world_centroid = homography.dot(new_centroid)
                    self.world_centroid = self.world_centroid[0], self.world_centroid[1]
                    cv2.putText(img_np_detect, str(round(self.world_centroid[0],2)) +','+ str(round(self.world_centroid[1],2)), centroid, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                rects.append([box, centroid, self.world_centroid, angle, detections['detection_classes'][i]])
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
        detection_result = np.bitwise_and(color_frame,box_mask)
        # detection_result = box_mask
        return img_np_detect, detection_result, rects

class robot_control:
    def __init__(self, Pick_place_dict, paths, files, checkpt):
        self.Pick_place_dict = Pick_place_dict
        self.paths = paths
        self.files = files
        self.checkpt = checkpt
    def get_nodes(self):
        self.Start_Prog = self.client.get_node('ns=3;s="HMIKuka"."robot"."example"."pickPlace"."command"."start"')
        self.Abort_Prog = self.client.get_node('ns=3;s="HMIKuka"."robot"."powerRobot"."command"."abort"')
        self.Rob_Stopped = self.client.get_node('ns=3;s="InstKukaControl"."instAutomaticExternal"."ROB_STOPPED"')
        self.Gripper_State = self.client.get_node('ns=3;s="gripper_control"')

        self.Act_Pos_X = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."X"')
        self.Act_Pos_Y = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Y"')
        self.Act_Pos_Z = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Z"')
        self.Act_Pos_A = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."A"')
        self.Act_Pos_B = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."B"')
        self.Act_Pos_C = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."C"')
        self.Act_Pos_Turn = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Turn"')
        self.Act_Pos_Status = self.client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Status"')

        self.PrePick_Pos_X = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."X"')
        self.PrePick_Pos_Y = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Y"')
        self.PrePick_Pos_Z = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Z"')
        self.PrePick_Pos_A = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."A"')
        self.PrePick_Pos_B = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."B"')
        self.PrePick_Pos_C = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."C"')
        self.PrePick_Pos_Status = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Status"')
        self.PrePick_Pos_Turn = self.client.get_node('ns=3;s="InstPickPlace"."positions"[1]."E6POS"."Turn"')

        self.Pick_Pos_X = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."X"')
        self.Pick_Pos_Y = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Y"')
        self.Pick_Pos_Z = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Z"')
        self.Pick_Pos_A = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."A"')
        self.Pick_Pos_B = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."B"')
        self.Pick_Pos_C = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."C"')
        self.Pick_Pos_Status = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Status"')
        self.Pick_Pos_Turn = self.client.get_node('ns=3;s="InstPickPlace"."positions"[2]."E6POS"."Turn"')

        self.Place_Pos_X = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."X"')
        self.Place_Pos_Y = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Y"')
        self.Place_Pos_Z = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Z"')
        self.Place_Pos_A = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."A"')
        self.Place_Pos_B = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."B"')
        self.Place_Pos_C = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."C"')
        self.Place_Pos_Status = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Status"')
        self.Place_Pos_Turn = self.client.get_node('ns=3;s="InstPickPlace"."positions"[3]."E6POS"."Turn"')

        self.PrePick_Done =  self.client.get_node('ns=3;s="InstPickPlace"."instPrePickPos"."Done"')
        self.Place_Done =  self.client.get_node('ns=3;s="InstPickPlace"."instPlacePos"."Done"')

    def show_boot_screen(self, message):
        cv2.namedWindow('Frame')
        boot_screen = np.zeros((960,1280))
        cv2.putText(boot_screen, message, (1280//2 - 150, 960//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Frame", boot_screen)
        cv2.waitKey(1)

    def connect_OPCUA_server(self):
        self.show_boot_screen('CONNECTING TO OPC UA SERVER...')
        password = "CIIRC"
        self.client = Client("opc.tcp://user:"+str(password)+"@10.35.91.101:4840/")
        self.client.connect()
        print("client connected")

    def change_trajectory(self, x, y, rot, packet_type):

        self.PrePick_Pos_X.set_value(ua.DataValue(ua.Variant(x, ua.VariantType.Float)))
        self.PrePick_Pos_Y.set_value(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        self.PrePick_Pos_Z.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['z'], ua.VariantType.Float)))
        self.PrePick_Pos_A.set_value(ua.DataValue(ua.Variant(rot, ua.VariantType.Float)))
        self.PrePick_Pos_B.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['b'], ua.VariantType.Float)))
        self.PrePick_Pos_C.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['c'], ua.VariantType.Float)))
        self.PrePick_Pos_Status.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        self.PrePick_Pos_Turn.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        self.Pick_Pos_X.set_value(ua.DataValue(ua.Variant(x, ua.VariantType.Float)))
        self.Pick_Pos_Y.set_value(ua.DataValue(ua.Variant(y, ua.VariantType.Float)))
        self.Pick_Pos_Z.set_value(ua.DataValue(ua.Variant(3.0, ua.VariantType.Float)))
        self.Pick_Pos_A.set_value(ua.DataValue(ua.Variant(rot, ua.VariantType.Float)))
        self.Pick_Pos_B.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['b'], ua.VariantType.Float)))
        self.Pick_Pos_C.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['c'], ua.VariantType.Float)))
        self.Pick_Pos_Status.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['status'], ua.VariantType.Int16)))
        self.Pick_Pos_Turn.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['pick_pos_base'][0]['turn'], ua.VariantType.Int16)))

        self.Place_Pos_X.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['x'], ua.VariantType.Float)))
        self.Place_Pos_Y.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['y'], ua.VariantType.Float)))
        self.Place_Pos_Z.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['z'], ua.VariantType.Float)))
        self.Place_Pos_A.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['a'], ua.VariantType.Float)))
        self.Place_Pos_B.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['b'], ua.VariantType.Float)))
        self.Place_Pos_C.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['c'], ua.VariantType.Float)))
        self.Place_Pos_Status.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['status'], ua.VariantType.Int16)))
        self.Place_Pos_Turn.set_value(ua.DataValue(ua.Variant(self.Pick_place_dict['place_pos'][packet_type]['turn'], ua.VariantType.Int16)))
    
        time.sleep(0.5)
    def compute_gripper_rot(self, angle):
        rot = 90.0 - abs(angle)
        return rot
        
    def gripper_gesture_control(self,detector, cap, show = False):
        
        if show:
            success, img = cap.read()
            hands, img = detector.findHands(img)  # with draw
                # hands = detector.findHands(img, draw=False)  # without draw
            if hands:
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"]  # List of 21 Landmark points
                bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
                centerPoint1 = hand1['center']  # center of the hand cx,cy
                handType1 = hand1["type"]  # Handtype Left or Right

                fingers1 = detector.fingersUp(hand1)

                if len(hands) == 2:
                    # Hand 2
                    hand2 = hands[1]
                    lmList2 = hand2["lmList"]  # List of 21 Landmark points
                    bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                    centerPoint2 = hand2['center']  # center of the hand cx,cy
                    handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                    fingers2 = detector.fingersUp(hand2)

                    # Find Distance between two Landmarks. Could be same hand or different hands
                    length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
                    length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
                    if length <=30.0:
                        self.Gripper_State.set_value(ua.DataValue(False))
                        time.sleep(0.1)
                    if length >=100.0:
                        self.Gripper_State.set_value(ua.DataValue(True))
                        time.sleep(0.1)
            cv2.imshow("Gestures", img)
        else:
            cv2.destroyWindow("Gestures")

    def objects_update(self,objects,image):
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] , centroid[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)
    def main_packet_detect(self):
        self.show_boot_screen('STARTING NEURAL NET...')
        warn_count = 0
        a = 0
        b = 0
        d = 2.61
        # d = 3
        bbox = True
        ct = CentroidTracker()    
        dc = DepthCamera()    
        pack_detect = packet_detector(self.paths, self.files, self.checkpt)
        homography = None
        while True:
            ret, depth_frame, color_frame, colorized_depth = dc.get_frame()
            
            color_frame = color_frame[:,240:1680]
            # color_frame = cv2.resize(color_frame, (640,480))
            height, width, depth = color_frame.shape[0],color_frame.shape[1],color_frame.shape[2]
            
            apriltag = processing_apriltag(None, color_frame, None)
            try:
                color_frame = apriltag.detect_tags()
                homography = apriltag.compute_homog()

                is_marker_detect= type(homography).__module__ == np.__name__ or homography == None
                if is_marker_detect:
                    warn_count = 0
                    # print(homography)
                    
            except:
            #Triggered when no markers are in the frame:
                warn_count += 1
                if warn_count == 1:
                    print("[INFO]: Markers out of frame or moving.")
                pass

            # color_frame = cv2.convertScaleAbs(color_frame, alpha=a, beta=b)
            # print(a,b,d)
            
            depth_frame = depth_frame[90:400,97:507]
            depth_frame = cv2.resize(depth_frame, (width,height))

            # heatmap = cv2.applyColorMap(np.uint8(depth_frame*d), cv2.COLORMAP_TURBO)

            heatmap = colorized_depth
            heatmap = heatmap[90:400,97:507,:]
            heatmap = cv2.resize(heatmap, (width,height))
            
            img_np_detect, result, rects = pack_detect.deep_detector(color_frame, depth_frame, homography, bnd_box = bbox)
            
            objects = ct.update(rects)
            self.objects_update(objects, img_np_detect)

            cv2.circle(img_np_detect, (int(width/2),int(height/2) ), 4, (0, 0, 255), -1)
            added_image = cv2.addWeighted(img_np_detect, 0.8, heatmap, 0.3, 0)
            
            cv2.imshow("Frame", cv2.resize(added_image, (1280,960)))
            # cv2.imshow("result", result)
            # cv2.imshow('object detection', cv2.resize(img_np_detect, (1280,960)))
            # cv2.imshow("Heatmap",cv2.resize(heatmap, (1280,960)))
            # cv2.imshow("Color", color_frame)
            
            key = cv2.waitKey(1)
            if key== ord('w'):
                a+=0.1
            if key== ord('s'):
                a-=0.1
            if key== ord('a'):
                b+=1
            if key== ord('d'):
                b-=1
            if key== ord('z'):
                d+=2
            if key== ord('x'):
                d-=2
            if key== ord('b'):
                if bbox == False:
                    bbox = True
                else:
                    bbox = False
            if key== 27:
                # cv2.destroyAllWindows()
                break
        print(rects)
        return added_image , rects

    def main_robot_control(self):
        detected_img, rects = self.main_packet_detect()

        self.connect_OPCUA_server()

        world_centroid = rects[0][2]
        packet_x = round(world_centroid[0] * 10.0, 2)
        packet_y = round(world_centroid[1] * 10.0, 2)
        angle = rects[0][3]
        gripper_rot = self.compute_gripper_rot(angle)
        packet_type = rects[0][4]

        self.get_nodes()

        frame_num = -1
        bpressed = 0
        dc = DepthCamera()
        gripper_ON = self.Gripper_State.get_value()
        cap = cv2.VideoCapture(1)
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        show_gestures = False
        while True:
            start = self.Start_Prog.get_value()
            rob_stopped = self.Rob_Stopped.get_value()
            abort = self.Abort_Prog.get_value()

            x_pos = self.Act_Pos_X.get_value()
            y_pos = self.Act_Pos_Y.get_value()
            z_pos = self.Act_Pos_Z.get_value()
            a_pos = self.Act_Pos_A.get_value()
            b_pos = self.Act_Pos_B.get_value()
            c_pos =self.Act_Pos_C.get_value()
            status_pos = self.Act_Pos_Status.get_value()
            turn_pos = self.Act_Pos_Turn.get_value()

            prePick_done = self.PrePick_Done.get_value()
            place_done = self.Place_Done.get_value()

            # print(start, x_pos, y_pos, z_pos, a_pos, b_pos, c_pos)

            ret, depth_frame, color_frame, colorized_depth = dc.get_frame()
            color_frame = color_frame[:,240:1680]
            frame_num += 1
            height, width, depth = color_frame.shape[0],color_frame.shape[1],color_frame.shape[2]
            # color_frame = cv2.convertScaleAbs(color_frame, alpha=1.2, beta=10)
            
            self.gripper_gesture_control(detector, cap, show = show_gestures)
            
            x_pos = round(x_pos,2)
            y_pos = round(y_pos,2)
            z_pos = round(z_pos,2)
            a_pos = round(a_pos,2)
            b_pos = round(b_pos,2)
            c_pos = round(c_pos,2)

            cv2.circle(color_frame, (int(width/2),int(height/2) ), 4, (0, 0, 255), -1)
            cv2.putText(color_frame,'x:'+ str(x_pos),(60,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'y:'+ str(y_pos),(60,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'z:'+ str(z_pos),(60,70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'a:'+ str(a_pos),(60,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'b:'+ str(b_pos),(60,110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'c:'+ str(c_pos),(60,130),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'Status:'+ str(status_pos),(60,150),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_frame,'Turn:'+ str(turn_pos),(60,170),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            color_frame = cv2.addWeighted(color_frame, 0.8, detected_img, 0.4, 0)

            cv2.imshow("Frame", cv2.resize(color_frame, (1280,960)))
            cv2.imshow('Object detected', cv2.resize(detected_img, (1280,960)))

            key = cv2.waitKey(1)
            if key == 27:
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',abort)
                self.Abort_Prog.set_value(ua.DataValue(False))
                cap.release()
                self.client.disconnect()
                self.gripper_gesture_control(detector, cap, show = False)
                cv2.destroyWindow("Object detected")
                # cv2.destroyAllWindows()
                time.sleep(0.5)
                break

            if rob_stopped:
                if key == ord('b'):
                    bpressed += 1
                    if bpressed == 5:
                        self.change_trajectory(packet_x, packet_y, gripper_rot, packet_type)
                        self.Start_Prog.set_value(ua.DataValue(True))
                        print('Program Started: ',start)
                        self.Start_Prog.set_value(ua.DataValue(False))
                        time.sleep(0.5)
                        bpressed = 0
                elif key != ord('b'):
                    bpressed = 0

            if key == ord('o'):
                self.Gripper_State.set_value(ua.DataValue(False))
                time.sleep(0.1)

            if key == ord('i'):
                self.Gripper_State.set_value(ua.DataValue(True))
                time.sleep(0.1)
            
            if key == ord('g'):
                if show_gestures == False:
                    show_gestures = True
                else:
                    show_gestures = False

            if key == ord('a'):
                self.Abort_Prog.set_value(ua.DataValue(True))
                print('Program Aborted: ',abort)
                time.sleep(0.5)

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
check_point ='ckpt-3'

# CUSTOM_MODEL_NAME = 'my_ssd_mobnet_improved_1' 
# check_point ='ckpt-6'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME) 
}
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

Pick_place_dict = {
"home_pos":[{'x':697.1,'y':0.0,'z':260.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':43}],

"pick_pos_base": [{'x':368.31,'y':226.34,'z':34.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':43}],

# place on conveyor points
"place_pos":[{'x':1079.44,'y':276.21,'z':45.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
            {'x':1250,'y':276.21,'z':45.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
            {'x':1420.73,'y':276.21,'z':45.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
            {'x':1420.73,'y':276.21,'z':45.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42}]
          }
#place on boxes points
# "place_pos":[{'x':1704.34,'y':143.92,'z':295.65,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
#             {'x':944.52,'y':124.84,'z':177.56,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
#             {'x':1284.27,'y':145.21,'z':274.95,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42},
#             {'x':1284.27,'y':145.21,'z':274.95,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':42}]
#           }

if __name__ == '__main__':
    while True:
        rc = robot_control(Pick_place_dict, paths, files, check_point)
        rc.main_robot_control()