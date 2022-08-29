import cv2
import argparse
import numpy as np
from robot_cell.detection.market_items_detector import ItemsDetector
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLACT Detection.')
    parser.add_argument('--weight', default='neural_nets/torch_yolact/weights/best_30.4_res101_coco_340000.pth', type=str)
    parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
    parser.add_argument('--hide_mask', default=False, action='store_true', help='Hide masks in results.')
    parser.add_argument('--hide_bbox', default=False, action='store_true', help='Hide boxes in results.')
    parser.add_argument('--hide_score', default=False, action='store_true', help='Hide scores in results.')
    parser.add_argument('--cutout', default=False, action='store_true', help='Cut out each object and save.')
    parser.add_argument('--save_lincomb', default=False, action='store_true', help='Show the generating process of masks.')
    parser.add_argument('--no_crop', default=False, action='store_true',
                        help='Do not crop the output masks with the predicted bounding box.')
    parser.add_argument('--real_time', default=True, action='store_true', help='Show the detection results real-timely.')
    parser.add_argument('--visual_thre', default=0.3, type=float,
                        help='Detections with a score under this threshold will be removed.')
    detector = ItemsDetector(parser, None, None, None)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        color_frame = cap.read()[1]
        detected_img = detector.deep_item_detector(color_frame, None, None)
        cv2.imshow('Detection', detected_img)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            break