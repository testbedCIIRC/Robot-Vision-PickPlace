import os
import sys
import json
import random
import time
import pyrealsense2 as rs
import cv2 
import numpy as np
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

class packet_detector:
    def __init__(self,paths,files):
        self.paths = paths
        self.files = files
        self.world_centroid = None
        self.category_index = label_map_util.create_category_index_from_labelmap(self.files['LABELMAP'])
        configs = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model= self.detection_model)
        ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

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
        # img = cv2.drawContours(img, contours, -1, (0,255,0), 3,lineType = cv2.LINE_AA)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        ret, mask = cv2.threshold(gray, 60, 255, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 8500:
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
        xmin,xmax,ymin,ymax = None,None,None,None
        box_mask = np.zeros_like(color_frame)
        image_np = np.array(color_frame)
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
                rects.append([box,centroid])
                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0),5)
                distance = depth_frame[centroid[1], centroid[0]]
                cv2.putText(img_np_detect, "{} deg".format(round(angle, 1)), (centroid[0], centroid[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img_np_detect, "{}mm".format(distance), (centroid[0], centroid[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if homography is not None:
                    new_centroid = np.append(centroid,1)
                    self.world_centroid = homography.dot(new_centroid)
                    self.world_centroid = round(self.world_centroid[0],2), round(self.world_centroid[1],2)
                    cv2.putText(img_np_detect, str(self.world_centroid), centroid, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        img_np_detect, box_mask = self.compute_mask(img_np_detect,box_mask, box_array)
        # print('hi',len(box_array))                
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
        return img_np_detect, detection_result, rects, xmin,xmax,ymin,ymax
class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.enable_auto_exposure, False)
        rgb_sensor = profile.get_device().query_sensors()[1]
        rgb_sensor.set_option(rs.option.enable_auto_exposure, False)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, 2)
        depth_frame = hole_filling.process(depth_frame)

        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, 0)

        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None, None
        return True, depth_image, color_image, colorized_depth

    def release(self):
        self.pipeline.stop()

packet_type = 'kld_042' 
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
LABEL_MAP_NAME = 'label_map.pbtxt'
IMAGES_PATH = 'C:/Users/David/PythonProjects/TF_OJ/TFODCourse/Tensorflow/packet_dataset/'+ packet_type +'/'
paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME) 
}
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

if __name__ == '__main__':

    warn_count = 0
    a = 1.0
    b = 0.0
    d = 2.61
    # d = 3
    bbox = True
    frame_num = -1
    dc = DepthCamera()    
    pack_detect = packet_detector(paths, files)
    homography = None
    while True:
        ret, depth_frame, color_frame, colorized_depth = dc.get_frame()
        color_frame = color_frame[:,240:1680]
        color_frame = cv2.resize(color_frame, (640,480))
        color_frame_output = color_frame
        # color_frame = cv2.resize(color_frame, (1280,960))
        
        height, width, depth = color_frame.shape[0],color_frame.shape[1],color_frame.shape[2]
        # print(height,width )
        # color_frame = cv2.convertScaleAbs(color_frame, alpha=a, beta=b)
        # print(a,b,d)
        
        depth_frame = depth_frame[94:394,32:572]
        depth_frame = cv2.resize(depth_frame, (width,height))
        
        img_np_detect, result, rects, xmin,xmax,ymin,ymax = pack_detect.deep_detector(color_frame, depth_frame, homography, bnd_box = bbox)
        if not (xmin==None or xmax==None or ymin==None or ymax==None):
            frame_num += 1
            print(frame_num)
            cv2.putText(img_np_detect,'Frame:'+ str(frame_num),(60,30),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
            img_name = packet_type +'_'+str(frame_num)

            # img_path = os.path.join('packet_dataset',packet_type,img_name)
            
            # cv2.imwrite(img_path + ".jpg", color_frame_output)
            # xml_file = open(img_path + '.xml','w' )
            # xml_file.write('<annotation>\n')
            # xml_file.write('	<folder>'+packet_type+'</folder>\n')
            # xml_file.write('	<filename>'+img_name+ ".jpg"+'</filename>\n')
            # xml_file.write('	<path>'+img_path.replace('/','\\')+ ".jpg"+'</path>\n')
            # xml_file.write('	<source>\n		<database>Unknown</database>\n	</source>\n	<size>\n' )
            # xml_file.write('		<width>'+str(width)+'</width>\n' )
            # xml_file.write('		<height>'+str(height)+'</height>\n' )
            # xml_file.write('		<depth>'+str(depth)+'</depth>\n' )
            # xml_file.write('	</size>\n	<segmented>0</segmented>\n	<object>\n')
            # xml_file.write('		<name>'+packet_type+'</name>\n')
            # xml_file.write('		<pose>Unspecified</pose>\n')
            # xml_file.write('		<truncated>0</truncated>\n')
            # xml_file.write('		<difficult>0</difficult>\n		<bndbox>\n')
            # xml_file.write('			<xmin>'+str(int(xmin))+'</xmin>\n')
            # xml_file.write('			<ymin>'+str(int(ymin))+'</ymin>\n')
            # xml_file.write('			<xmax>'+str(int(xmax))+'</xmax>\n')
            # xml_file.write('			<ymax>'+str(int(ymax))+'</ymax>\n')
            # xml_file.write('		</bndbox>\n	</object>\n</annotation>\n')
            # xml_file.close()
        cv2.imshow('object detection', cv2.resize(img_np_detect, (1280,960)))
        # cv2.imshow("Frame", color_frame)
        cv2.imshow("Frame", color_frame_output)
        time.sleep(0.3)
        k = cv2.waitKey(1)
        if k == ord('w'):
            a+=1
        if k == ord('s'):
            a-=1
        if k == ord('a'):
            b+=1
        if k == ord('d'):
            b-=1
        if k == ord('z'):
            d+=2
        if k == ord('x'):
            d-=2
        if k == ord('b'):
            if bbox == False:
                bbox = True
            else:
                bbox = False
        if k == 27:
            cv2.destroyAllWindows()
            break
            