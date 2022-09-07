import cv2
import argparse
import numpy as np
from robot_cell.detection.market_items_detector import ItemsDetector
from robot_cell.detection.apriltag_detection import ProcessingApriltag
from robot_cell.detection.realsense_depth import DepthCamera
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLACT Detection.')
    parser.add_argument('--weight', default='neural_nets/torch_yolact/weights/best_31.9_swin_tiny_coco_308000.pth', type=str)
    # parser.add_argument('--weight', default='neural_nets/torch_yolact/weights/best_30.4_res101_coco_340000.pth', type=str)
    # parser.add_argument('--weight', default='neural_nets/torch_yolact/weights/best_28.8_res50_coco_340000.pth', type=str)
    parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
    parser.add_argument('--hide_mask', default=True, action='store_true', help='Hide masks in results.')
    parser.add_argument('--hide_bbox', default=False, action='store_true', help='Hide boxes in results.')
    parser.add_argument('--hide_score', default=False, action='store_true', help='Hide scores in results.')
    parser.add_argument('--cutout', default=False, action='store_true', help='Cut out each object and save.')
    parser.add_argument('--save_lincomb', default=False, action='store_true', help='Show the generating process of masks.')
    parser.add_argument('--no_crop', default=False, action='store_true',
                        help='Do not crop the output masks with the predicted bounding box.')
    parser.add_argument('--real_time', default=True, action='store_true', help='Show the detection results real-timely.')
    parser.add_argument('--visual_thre', default=0.8, type=float,
                        help='Detections with a score under this threshold will be removed.')
    detector = ItemsDetector(parser, None, None, None)
    dc = DepthCamera()
    apriltag = ProcessingApriltag()
    apriltag.load_world_points('config/conveyor_points.json')
    while True:
        t1 = time.time()
        success, depth_frame, rgb_frame, colorized_depth = dc.get_frames()
        # rgb_frame = rgb_frame[:, 240:1680]
        apriltag.detect_tags(rgb_frame)
        # homography = apriltag.compute_homog()
        image_frame = rgb_frame.copy()
        print(rgb_frame.shape)
        if not success:
            continue
        frame_height, frame_width, frame_channel_count = rgb_frame.shape

        # color_frame = cap.read()[1]
        detected_img, packets = detector.deep_item_detector(rgb_frame, None, None,image_frame)
        # detected_img2, packets2 = detector.deep_item_detector(color_frame, None, None)
        detected_img = apriltag.draw_tags(detected_img)
        t2 = time.time()
        print(f"Delta t {t2-t1}, FPS = {1/(t2-t1)}")
        # print(len(packets), len(packets2))
        cv2.imshow('Detection', cv2.resize(detected_img,(frame_width // 2, frame_height // 2)))
        # cv2.imshow('Detection2', detected_img2)
        key = cv2.waitKey(1)
        if key == 27:
            dc.release()
            break