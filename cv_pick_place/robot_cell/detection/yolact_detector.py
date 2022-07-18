from dataclasses import make_dataclass
from email.mime import image
import os
from pydoc import ModuleScanner
from tkinter.messagebox import NO
import cv2
import numpy as np
import sys
# sys.path.append("..")
# sys.path.append(".")
# from cv_pick_place.robot_cell.packet.packet_object import Packet
import tensorflow as tf

from yolact.yolact import Yolact
from robot_cell.detection.yolact_config import *

from robot_cell.packet.packet_object import Packet

class YolactDetector:
    def __init__(self, parameters:dict):
        """
        Initializes detector of YOLACT model, load weights and all preparations
        for predictions

        Parameters:
        params (dict): Dictionary with parameters, see main() for usage

        """
        # Model settings and init
        self.task = parameters.get('task')

        model_params = {
            "backbone": BACKBONE,
            "fpn_channels": FPN_CHANNELS,
            "num_class": NUM_CLASSES[self.task],
            "num_mask": NUM_MASK,
            "anchor_params": ANCHOR[self.task],
            "detect_params": DETECT[self.task]
        }

        self.model = Yolact(**model_params)
        input_arr = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        self.model(input_arr)

        weights_path = parameters.get('weights_path')
        if weights_path is None:
            weights_path = 'cv_pick_place\yolact\weights\weights_packets.h5'
            # TODO Tady to asi nějak ošetřit aby to pak fungovalo
            print(f"[WARN]: Weights were not specified using defaults FOR NOW from: \n\t{weights_path}")
                
        self.model.load_weights(weights_path)

        # Postprocessing
        self.confidance_threshold = parameters.get('confidance_threshold', 0.5)

        # Visualization settings
        self.font_face = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_color = [0, 0, 0]
        self.colors = COLORS
        self.class_names = CLASS_NAMES.get(self.task)

        self.show = parameters.get('show', False)
        self.save = parameters.get('save', False)
        self.save_path = parameters.get('save_path', False)
        self.frame_num = parameters.get('frame_num')
        # Creates new folder if savig is desired
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)

        self.save_path = parameters.get('save_path', False)
        self.frame_num = parameters.get('frame_num')

    def sanitize_coordinates(self, x1, x2, img_size, padding=0):
        x1 = tf.minimum(x1, x2)
        x2 = tf.maximum(x1, x2)
        x1 = tf.clip_by_value(x1 - padding, clip_value_min=0., clip_value_max=1000000.)
        x2 = tf.clip_by_value(x2 + padding, clip_value_min=0., clip_value_max=tf.cast(img_size, tf.float32))
        return x1, x2


    def crop(self, pred, boxes):
        # pred [num_obj, 138, 138], gt [num_bboxes, 4]
        # sanitize coordination (make sure the bboxes are in range 0 <= x, y <= image size)
        shape_pred = tf.shape(pred)
        pred_w = shape_pred[0]
        pred_h = shape_pred[1]

        xmin, xmax = self.sanitize_coordinates(boxes[:, 0], boxes[:, 2], pred_w, padding=1)
        ymin, ymax = self.sanitize_coordinates(boxes[:, 1], boxes[:, 3], pred_h, padding=1)

        rows = tf.broadcast_to(tf.range(pred_w)[None, :, None], shape_pred)
        cols = tf.broadcast_to(tf.range(pred_h)[:, None, None], shape_pred)

        xmin = xmin[None, None, :]
        ymin = ymin[None, None, :]
        xmax = xmax[None, None, :]
        ymax = ymax[None, None, :]

        mask_left = (rows >= tf.cast(xmin, cols.dtype))
        mask_right = (rows <= tf.cast(xmax, cols.dtype))
        mask_bottom = (cols >= tf.cast(ymin, rows.dtype))
        mask_top = (cols <= tf.cast(ymax, rows.dtype))

        crop_mask = tf.math.logical_and(tf.math.logical_and(mask_left, mask_right),
                                        tf.math.logical_and(mask_bottom, mask_top))
        return pred * tf.cast(crop_mask, tf.float32)

    def postprocess(self, detection, w, h, batch_idx,
                    intepolation_mode: str = "bilinear", crop_mask: bool = True,
                    score_threshold: float = 0.5) -> tuple[np.ndarray]:
        """
        Postprocessing of the results from prediction, interpolates image to 
        original size, recalulationof coords into pixel

        Parameters:
        detection (dict): Dictionary with detectios
        w (int): Original image width
        h (int): Original image height
        batch_idx (int): Idx of batch
        interpolation_mode (str): Method used for recising
        crop_mask (bool): 
        score_threshold (float): Confidance score threshold for detection to be
            considered acceptable

        Returns:
        tuple[np.ndarrays]: predictions(class, confidence, bbox, mask)
        """
        dets = detection[batch_idx]
        dets = dets['detection']
        if dets is None: 
            return None, None, None, None

        keep = tf.squeeze(tf.where(dets['score']> self.confidance_threshold))
        # TODO Zkouknout co to vlastně dělá
        # Asi prochází všechny klíče kromě 'proto'
        for k in dets.keys():
            if k != 'proto':
                dets[k] = tf.gather(dets[k], keep)
        
        if tf.size(dets['score']) == 0:
            return None, None, None, None

        classes = dets['class']
        boxes = dets['box']
        scores = dets['score']
        masks = dets['mask']
        proto_pred = dets['proto']

        if tf.rank(masks) == 1:
            masks = tf.expand_dims(masks, axis=0)
            classes = tf.expand_dims(classes, axis=0)
            boxes = tf.expand_dims(boxes, axis=0)
            scores = tf.expand_dims(scores, axis=0)

        pred_mask = tf.linalg.matmul(proto_pred, masks, transpose_a=False, transpose_b=True)
        pred_mask = tf.nn.sigmoid(pred_mask)

        if crop_mask:
            masks = self.crop(pred_mask, boxes * float(tf.shape(pred_mask)[0] / IMG_SIZE))
        masks = tf.transpose(masks, perm=[2, 0, 1])

        # intepolate to original size
        masks = tf.image.resize(tf.expand_dims(masks, axis=-1), [w, h],
                                method=intepolation_mode)
        # binarized the mask
        masks = tf.cast(masks + 0.5, tf.int64)
        masks = tf.squeeze(tf.cast(masks, tf.float32))  
        # tf.print("masks after postprecessing", tf.shape(masks))

        # Conversion to numpy for easier math operation
        classes, scores, boxes, masks = classes.numpy(), scores.numpy(), boxes.numpy(), masks.numpy()

        # Resizing of the predicted boxes
        boxes /= IMG_SIZE
        boxes[:, [0,2]] *= self.orig_img_size[1]
        boxes[:, [1,3]] *= self.orig_img_size[0]

        return classes, scores, boxes, masks

    def load_image(self, img_path: str) -> np.ndarray:
        """
        Load image from image_path

        Parameters:
        img_path(str): path to image

        Returns: 
        np.ndarray: Loaded image
        """
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = np.array(img)
        return img
    
    def prepare_image(self, img: np.ndarray) -> tf.Tensor:
        """
        Prepares image for the detection by model by converting it to tf tensor
        
        Parameters:
        img (np.ndarray): image to be prepared

        Returns:
        (tf.tensor): image prepared for detection
        """
        self.orig_img_size = img.shape

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = np.asarray(img)

        img = tf.convert_to_tensor(img,  dtype=tf.float32)
        img = tf.expand_dims(img, 0)
        return img

    def detect_fn(self, image:tf.Tensor) -> tuple[np.ndarray]:
        """
        Detects objects in the image

        Parameters:
        image(tf.tensor) : image onto which prediction is supposed to be made
        
        Returns:
        tuple[np.ndarrays]: predictions(class, confidence, bbox, mask)
        
        """
        out = self.model(image, training=False)
        detection = self.model.detect(out)
        out = self.postprocess(detection, self.orig_img_size[0], self.orig_img_size[1], 0, "bilinear")
        return out

    def visualize_prediction(self, image:np.ndarray, predictions:tuple[np.ndarray],bnd_box:bool=True, segment:bool=True):
        # TODO Po vyzkoušení vymazat funkci je redundantní
        # Asi je to už v pohodě před nahrátím na github zrušit
        """
        Visualize prediction on image

        Parameters:
        image (), 
        prediction (tuple[np.ndarray]), 
        """
        cl, cf, bbox, masks = predictions
        masks = masks[None, :, :] if masks.shape[0]== IMG_SIZE else masks
        mask_final = np.zeros_like(masks[0][:, :, None])
        img = cv2.resize(image, (self.orig_img_size[1],self.orig_img_size[0]), interpolation=cv2.INTER_LINEAR)
        img = np.float32(img)/255.0
        # cv2.imshow("Original image", img)
        # cv2.waitKey(0)
        for i in range(bbox.shape[0]):
            b = bbox[i].astype(int)
            x1, y1, x2, y2 = b
            m = masks[i][:,:,None]
            # b bbox coord
            color = self.colors[int(cl[i] % len(self.colors))]
            text_orig = (int(x1), int(y1 - 3))
            text = f"{self.class_names[cl[i]]} {'%.2f' % round(cf[i], 2)}"
            text_w, text_h = cv2.getTextSize(text, self.font_face, self.font_scale, self.font_thickness)[0]
            # BBOX + text
            if bnd_box:
                cv2.rectangle(img,  (x1, y1), (x2, y2), color, 1)
                # TODO Plotting of class and confidance could be separated
                cv2.rectangle(img, (x1, y1), (int(x1 + text_w), int(y1 - text_h - 4)), color, -1)
                cv2.putText(img, text, text_orig, self.font_face, self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)
            
            mask_final = mask_final + np.concatenate((m * color[0] / 255.0, m * color[1] / 255.0, m*color[2] / 255.0), axis=-1)

        dst = np.zeros_like(img)
        if segment:
            cv2.addWeighted(img, 0.7, mask_final, 0.3, 0, dst)
        else:
            dst = img

        if self.save:
            cv2.imwrite(os.path.join(self.save_path, f"image{self.frame_num}.jpg"), dst*255)
            print("[INFO]: Saved image to ", self.save_path)
        else:
            cv2.imshow("Detection image", dst)
            cv2.waitKey(0)
        return dst

    def determine_packet_class(self, image, mask, homography_deterinant) -> int:
        """
        Determines the class of detected packet, Used only when only
        packets are detected

        Parameters:
        image (np.ndarray): Image
        mask (np.ndarray): Binary mask with detected packet
        homography_determinant (float): Determinant of homography

        Returns:
        int: class index of the detected packet 

        """
        area = np.count_nonzero(mask) * homography_deterinant
        # print(area)
        # TODO: There must be something to determine the class of the packets based on size of area and color of the package
        # As the packet is brown it should have lesser value in blue channel than the white one
        # These values are taken form threshold detector however it uses different computation of area, 
        # though it should be close to the mask area, mask is however slightly bigger

        class_idx = 0
        if 110 > area > 90:# Small white
            class_idx = 1
        elif 180 > area > 160: # Middle white and brown
            # TODO: Here should be something about difference between class 2 and 3
            class_idx = 2
        elif 380 > area > 350: # Large white
            clas_idx = 0
        


    def detect_packet(self, color_frame:np.ndarray, encoder_pos:float,
                      homography:np.ndarray, bnd_box:bool= True,
                      segment:bool = True, image_frame:np.ndarray = None
                      ) -> tuple[np.ndarray, list[Packet]]:
        """
        Detects packets in the frame. 
        If none are found returns image and empty list. 

        Parameters:
        color_frame (np.ndarray): Input image frame
        encoder_pos (float): current encoder position
        homography (np.ndarray): Homography matrix
        bnd_box (bool): Switch for enabling bounding box visualization
        segment (bool): Switch for enabling segmentation mask visualization
        image_frame (np.ndarray): Image with output visualization, if not 
            provided copy of color_frame is used 

        Returns:
        tuple: Image with visualization, list with Packet class
        """
        if homography is not None:
            homography_det = np.linalg.det(homography[0:2, 0:2])

        img_tf = self.prepare_image(color_frame)
        predictions = self.detect_fn(img_tf)
        
        if image_frame is None or not image_frame.shape == color_frame.shape:
            img_draw = color_frame.copy()
        else:
            img_draw = image_frame
        
        packet_list = []
        cl, cf, bbox, masks = predictions
        # masks = masks[None, :, :] if masks.shape[0] == IMG_SIZE else masks
        if cl is None:
            # TODO: As it sits right now it returs input image and empty list
            # Zeptat se Jirky co vracet
            print(f"[INFO]: Detected nothing in the image")
        else:
            masks = masks[None, :, :] if len(masks.shape) == 2 else masks
            # Tak tady pak když se vrátí jenom jedna věc
            mask_final = np.zeros_like(masks[0][:, :, None])
            img = cv2.resize(img_draw, (self.orig_img_size[1],self.orig_img_size[0]), interpolation=cv2.INTER_LINEAR)
            img = np.float32(img)/255.0

            for i in range(bbox.shape[0]):
                klass = 0
                b = bbox[i].astype(int)
                x1, y1, x2, y2 = b
                m = masks[i][:,:,None]
                vals, count = np.unique(m, return_counts=True)
                
                cx,cy = (x2+x1)/2,(y2+y1)/2
                centroid = (int(cx),int(cy))
                # print(centroid)
                if self.task == 'packets' and homography is not None:
                   klass = self.determine_packet_class(image_frame, m, homography_det)
                # w = float((x2 - x1)) / self.orig_img_size[1]
                # h = float((y2 - y1)) / self.orig_img_size[0]

                w = int((x2 - x1))
                h = int((y2 - y1))
                packet = Packet(
                    # Je možné že se zruší/předělá init, protože se to nepoužívá
                    box = [], # může být 
                    centroid = centroid, # tupple x,y
                    # centroid_depth= depth_frame[centroid[1], centroid[0]], # Obsolete not used in any way
                    pack_type=  klass,
                    angle = 0,
                    xmax= x2, xmin= x1,
                    ymax= y2, ymin= y1,
                    width= w, height= h, 
                    encoder_position= encoder_pos
                )
                packet.set_type(klass)
                packet.set_centroid(int(cx), int(cy), homography, encoder_pos)
                packet.set_bounding_size(w,h,homography)

                packet_list.append(packet)

                # Color and text setting
                color = self.colors[int(cl[i] % len(self.colors))]
                text_orig = (int(x1), int(y1 - 3))
                text = f"{self.class_names[cl[i]]} {'%.2f' % round(cf[i], 2)}"
                text_w, text_h = cv2.getTextSize(text, self.font_face, self.font_scale, self.font_thickness)[0]
                # BBOX + text
                if bnd_box or segment:
                    cv2.rectangle(img, (x1, y1), (int(x1 + text_w), int(y1 - text_h - 4)), color, -1)
                    cv2.putText(img, text, text_orig, self.font_face, self.font_scale, self.text_color,
                                self.font_thickness, cv2.LINE_AA)
                if bnd_box:
                    cv2.rectangle(img,  (x1, y1), (x2, y2), color, 1)
                    # cv2.rectangle(img, (x1, y1), (int(x1 + text_w), int(y1 - text_h - 4)), color, -1)
                    # cv2.putText(img, text, text_orig, self.font_face, self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)
                
                if segment:
                    mask_final = mask_final + np.concatenate((m * color[0] / 255.0, m * color[1] / 255.0, m*color[2] / 255.0), axis=-1)

            dst = np.zeros_like(img)
            if segment:
                cv2.addWeighted(img, 0.7, mask_final, 0.3, 0, dst)
            else:
                dst = img

            if self.save:
                cv2.imwrite(os.path.join(self.save_path, f"image{self.frame_num}.jpg"), dst*255)
                print(f"[INFO]: Saved image to {self.save_path}")
            if self.show:
                cv2.imshow("Detection image", dst)
                cv2.waitKey(0)

        self.frame_num += 1
        return img_draw, packet_list



def main():
    # Demo how to work with this detection
    # Select task, packets/types
    task = 'packets'
    # task = 'packet_types'
    
    # Setting parameters for the model
    parameters = {
        "save": False,
        "show": True,
        "save_path": "test\predictions",
        "confidence_threshold": 0.5,
        "task": task,
        "frame_num": 0}

    # Initializing model
    detector =  YolactDetector(parameters=parameters)
    print(f"[INFO]: Detector (NN) initialized")
    # Load image
    # path_to_img = "G:\VSProgramming\Robot-Vision-PickPlace\TensorflowYOLACT\test\imgs\image7.jpg"
    path_to_img = "cv_pick_place\yolact\\test\imgs\image7.jpg"
    image = detector.load_image(path_to_img) # ADD SOMETHING FOR TESTING
    print(f"[INFO]: Image loaded")

    detector.detect_packet(image, encoder_pos=None, homography=None, bnd_box=False, segment=True)
    print(f"[INFO]: Prediction made")
    
    # Second IMG JUST FOR visualization
    path_to_img = "cv_pick_place\yolact\\test\imgs\image4.jpg"
    image = detector.load_image(path_to_img) # ADD SOMETHING FOR TESTING
    print(f"[INFO]: Image loaded")

    detector.detect_packet(image, encoder_pos=None, homography=None, bnd_box=True, segment=False)
    print(f"[INFO]: Prediction made")

    path_to_img = "cv_pick_place\yolact\\test\imgs\\black.jpg"
    image = detector.load_image(path_to_img) # ADD SOMETHING FOR TESTING
    print(f"[INFO]: Image loaded")

    detector.detect_packet(image, encoder_pos=None, homography=None, bnd_box=True, segment=True)
    print(f"[INFO]: Prediction made")

    # save_folder = "models"
    # os.makedirs(save_folder, exist_ok=True)
    # model_save = os.path.join(save_folder, "packet.h5")
    # input_arr = np.ones((1, IMG_SIZE, IMG_SIZE, 3))
    # print(f"[INFO]: Trying to prepare the model")
    # detector.model._set_inputs(input_arr) # add this line

    # detector.model.compute_output_shape(input_shape=(1, IMG_SIZE, IMG_SIZE, 3))

    # save_folder = "models"
    # # os.makedirs(save_folder, exist_ok=True)
    # # model_save = os.path.join(save_folder, "packet")
    # # self.model.save(model_save, save_format="tf")
    # # self.model.load_weights(weights_path)
    # model_save2 = os.path.join(save_folder, "packetww2")
    # print(f"[INFO]: Trying to save the model")
    # detector.model.save(model_save2)
if __name__ == "__main__":
    main()
