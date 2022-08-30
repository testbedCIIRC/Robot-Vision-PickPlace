import os
import re
import argparse
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from robot_cell.packet.packet_object import Packet
from neural_nets.torch_yolact.yolact import Yolact
from neural_nets.torch_yolact.config import get_config, COLORS
from neural_nets.torch_yolact.augmentations import val_aug
from neural_nets.torch_yolact.box_utils import match, crop, make_anchors, box_iou

class ItemsDetector:
    """
    Class for detecting packets using neural network.
    """

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        paths: dict,
        files: dict,
        checkpt: str,
        max_detect: int = 1,
        detect_thres: float = 0.7,
    ):
        """
        PacketDetector object constructor.

        Args:
            paths (dict): Dictionary with annotation and checkpoint paths.
            files (dict): Dictionary with pipeline and config paths.
            checkpt (str): Name of training checkpoint to be restored.
            max_detect (int): Maximal ammount of concurrent detections in an image.
            detect_thres (float): Minimal confidence for detected object to be labeled as a packet.
        """

        args = parser.parse_args()
        prefix = re.findall(r'best_\d+\.\d+_', args.weight)[0]
        suffix = re.findall(r'_\d+\.pth', args.weight)[0]
        args.cfg = args.weight.split(prefix)[-1].split(suffix)[0]
        cfg = get_config(args, mode='detect')
        self.cfg = cfg
        self.net = Yolact(cfg)
        self.net.load_weights(cfg.weight, cfg.cuda)
        self.net.eval()

        if cfg.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.net = self.net.cuda()


    def fast_nms(self, box_thre, coef_thre, class_thre, cfg):
        class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]

        idx = idx[:, :cfg.top_k]
        class_thre = class_thre[:, :cfg.top_k]

        num_classes, num_dets = idx.size()
        box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
        coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

        iou = box_iou(box_thre, box_thre)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= cfg.nms_iou_thre)

        # Assign each kept detection to its corresponding class
        class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

        class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        class_nms, idx = class_nms.sort(0, descending=True)

        idx = idx[:cfg.max_detections]
        class_nms = class_nms[:cfg.max_detections]

        class_ids = class_ids[idx]
        box_nms = box_nms[idx]
        coef_nms = coef_nms[idx]

        return box_nms, coef_nms, class_ids, class_nms
    
    def nms(self, class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
        class_p = class_pred.squeeze()  # [19248, 81]
        box_p = box_pred.squeeze()  # [19248, 4]
        coef_p = coef_pred.squeeze()  # [19248, 32]
        proto_p = proto_out.squeeze()  # [138, 138, 32]

        if isinstance(anchors, list):
            anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)

        class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

        # exclude the background class
        class_p = class_p[1:, :]
        # get the max score class of 19248 predicted boxes
        class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

        # filter predicted boxes according the class score
        keep = (class_p_max > cfg.nms_score_thre)
        class_thre = class_p[:, keep]
        box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]

        # decode boxes
        box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                            anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)
        box_thre[:, :2] -= box_thre[:, 2:] / 2
        box_thre[:, 2:] += box_thre[:, :2]

        box_thre = torch.clip(box_thre, min=0., max=1.)

        if class_thre.shape[1] == 0:
            return None, None, None, None, None
        else:
            box_thre, coef_thre, class_ids, class_thre = self.fast_nms(box_thre, coef_thre, class_thre, cfg)
            return class_ids, class_thre, box_thre, coef_thre, proto_p
    
    def after_nms(self, ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None, img_name=None):
        if ids_p is None:
            return None, None, None, None

        if cfg and cfg.visual_thre > 0:
            keep = class_p >= cfg.visual_thre
            if not keep.any():
                return None, None, None, None

            ids_p = ids_p[keep]
            class_p = class_p[keep]
            box_p = box_p[keep]
            coef_p = coef_p[keep]

        masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t()))

        if not cfg or not cfg.no_crop:  # Crop masks by box_p
            masks = crop(masks, box_p)

        masks = masks.permute(2, 0, 1).contiguous()

        ori_size = max(img_h, img_w)
        # in OpenCV, cv2.resize is `align_corners=False`.
        masks = F.interpolate(masks.unsqueeze(0), (ori_size, ori_size), mode='bilinear', align_corners=False).squeeze(0)
        masks.gt_(0.5)  # Binarize the masks because of interpolation.
        masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

        box_p *= ori_size
        box_p = box_p.int()

        return ids_p, class_p, box_p, masks
    
    def draw_img(
        self, 
        ids_p, 
        class_p, 
        box_p, 
        mask_p, 
        img_origin, 
        cfg, 
        img_name=None, 
        fps=None):

        if ids_p is None:
            return img_origin

        if isinstance(ids_p, torch.Tensor):
            ids_p = ids_p.cpu().numpy()
            class_p = class_p.cpu().numpy()
            box_p = box_p.cpu().numpy()
            mask_p = mask_p.cpu().numpy()

        num_detected = ids_p.shape[0]

        img_fused = img_origin
        if not cfg.hide_mask:
            masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
            # The color of the overlap area is different because of the '%' operation.
            masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
            color_masks = COLORS[masks_semantic].astype('uint8')
            img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)
        
        scale = 0.6
        thickness = 1
        font = cv2.FONT_HERSHEY_DUPLEX

        if not cfg.hide_bbox:
            for i in reversed(range(num_detected)):
                x1, y1, x2, y2 = box_p[i, :]

                color = COLORS[ids_p[i] + 1].tolist()
                cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

                class_name = cfg.class_names[ids_p[i]]
                text_str = f'{class_name}: {class_p[i]:.2f}' if not cfg.hide_score else class_name

                text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
                cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
                cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
        return img_fused

    def get_item_from_mask(
        self, img: np.array, bbox:dict, mask: np.array, type: int, encoder_pos: float
    ) -> Packet:
        """
        Creates object inner rectangle given the mask from neural net.

        Args:
            img (np.array): image for drawing detections.
            bbox (dict): bounding box parameters.
            mask (np.array): mask produced by neural net.
            type (int): Type of the packet.
            encoder_pos (float): Position of the encoder.

        Returns:
            Packet: Created Packet object
        """
        centroid = bbox['centroid']
        ymin = bbox['ymin']
        ymax = bbox['ymax']
        xmin = bbox['xmin']
        xmax = bbox['xmax']
        angle = 0
        box = np.int64(
            np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        )
        contours, _  = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000:
                rectangle = cv2.minAreaRect(cnt)
                centroid = (int(rectangle[0][0]), int(rectangle[0][1]))
                box = np.int0(cv2.boxPoints(rectangle))
                angle = int(rectangle[2])

        cv2.polylines(img, [box], True, (255, 0, 0), 3)
        packet = Packet(
            box=box,
            pack_type=type,
            centroid=centroid,
            angle=angle,
            ymin=ymin,
            ymax=ymax,
            xmin=xmin,
            xmax=xmax,
            width=bbox['w'],
            height=bbox['h'],
            encoder_position=encoder_pos,
        )

        packet.set_type(type)
        packet.add_angle_to_average(angle)

        return packet

    def deep_item_detector(
        self,
        rgb_frame: np.ndarray,
        encoder_pos: float,
        draw_box: bool = True,
        image_frame: np.ndarray = None,
    ) -> tuple[np.ndarray, list[Packet]]:
        """
        Detects packets using Yolact model.

        Args:
            rgb_frame (np.ndarray): RGB frame in which packets should be detected.
            encoder_position (float): Position of the encoder.
            draw_box (bool): If bounding and min area boxes should be drawn.
            image_frame (np.ndarray): Image frame into which information should be drawn.

        Returns:
            np.ndarray: Image frame with information drawn into it.
            list[Packet]: List of detected packets.
        """
        scale = 0.6
        thickness = 1
        font = cv2.FONT_HERSHEY_DUPLEX
        detected = []
        img_h, img_w = rgb_frame.shape[0:2]
        frame_trans = val_aug(rgb_frame, self.cfg.img_size)

        frame_tensor = torch.tensor(frame_trans).float()
        if self.cfg.cuda:
            frame_tensor = frame_tensor.cuda()

        with torch.no_grad():
            class_p, box_p, coef_p, proto_p = self.net.forward(frame_tensor.unsqueeze(0))
        
        ids_p, class_p, box_p, coef_p, proto_p = self.nms(
            class_p, box_p, coef_p, proto_p, self.net.anchors, self.cfg
            )
        ids_p, class_p, boxes_p, masks_p = self.after_nms(
            ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, self.cfg
            )
        # img_np_detect = self.draw_img(ids_p, class_p, boxes_p, masks_p, rgb_frame, self.cfg)
        
        if ids_p is None:
            return rgb_frame,detected

        if isinstance(ids_p, torch.Tensor):
            ids_p = ids_p.cpu().numpy()
            class_p = class_p.cpu().numpy()
            boxes_p = boxes_p.cpu().numpy()
            masks_p = masks_p.cpu().numpy()

        num_detected = ids_p.shape[0]

        img_np_detect = rgb_frame
        if not self.cfg.hide_mask:
            masks_semantic = masks_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
            # The color of the overlap area is different because of the '%' operation.
            masks_semantic = masks_semantic.astype('int').sum(axis=0) % (self.cfg.num_classes - 1)
            color_masks = COLORS[masks_semantic].astype('uint8')
            img_np_detect = cv2.addWeighted(color_masks, 0.4, rgb_frame, 0.6, gamma=0)
        
        for i in reversed(range(num_detected)):
            xmin, ymin, xmax, ymax = boxes_p[i, :]
            w = float((xmax - xmin)) / img_w
            h = float((ymax - ymin)) / img_h
            cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
            centroid = (int(cx), int(cy))
            
            bbox = {
                'xmin':xmin, 
                'ymin':ymin,
                'xmax': xmax,
                'ymax': ymax,
                'centroid':centroid, 
                'w': w, 
                'h': h
            }
            
            if not self.cfg.hide_bbox:
                color = COLORS[ids_p[i] + 1].tolist()
                cv2.rectangle(img_np_detect, (xmin, ymin), (xmax, ymax), color, thickness)

                class_name = self.cfg.class_names[ids_p[i]]
                text_str = f'{class_name}: {class_p[i]:.2f}' if not self.cfg.hide_score else class_name

                text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
                cv2.rectangle(img_np_detect, (xmin, ymin), (xmin + text_w, ymin + text_h + 5), color, -1)
                cv2.putText(
                    img_np_detect, 
                    text_str, 
                    (xmin, ymin + 15), 
                    font, 
                    scale, 
                    (255, 255, 255), 
                    thickness, 
                    cv2.LINE_AA
                )

            packet = self.get_item_from_mask(
                img_np_detect, 
                bbox, masks_p[i], 
                int(ids_p[i]), 
                encoder_pos
            )
            detected.append(packet)
        return img_np_detect, detected

