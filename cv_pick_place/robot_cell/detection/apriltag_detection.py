import json
import cv2 
import numpy as np
class ProcessingApriltag:
	
	def __init__(self):
		"""
        ProcessingApriltag object constructor. Initializes data containers.
		
        """
		
		self.image_points = {}
		self.world_points = {}
		self.world_points_detect = []
		self.image_points_detect = []
		self.homography = None
	
	def load_original_points(self):
		"""
        Loads conveyor world points from the json file.
        
        """
		f = open('conveyor_points.json')
		# Dict of points in conveyor:
		self.world_points = json.load(f) 

	def compute_homog(self):
		"""
        Computes homography matrix using image and conveyor world points.

        Returns:
        numpy.ndarray: Homography matrix as numpy array.
        
        """
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

	def detect_tags(self, color_image):
		"""
        Detects and draws tags on a copy if the input image.
		
		Parameters:
        color_image (numpy.ndarray): Input image where apriltags are to be detected.

        Returns:
        numpy.ndarray: Image with apriltags.
        
        """
		rect1 = []
		image = color_image.copy()
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