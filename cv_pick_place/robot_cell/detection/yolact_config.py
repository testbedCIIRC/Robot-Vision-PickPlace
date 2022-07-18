COLORS = (
    (244, 67, 54),
    (233, 30, 99),
    (156, 39, 176),
    (103, 58, 183),
    (63, 81, 181),
    (33, 150, 243),
    (3, 169, 244),
    (0, 188, 212),
    (0, 150, 136),
    (76, 175, 80),
    (139, 195, 74),
    (205, 220, 57),
    (255, 235, 59),
    (255, 193, 7),
    (255, 152, 0),
    (255, 87, 34),
    (121, 85, 72),
    (158, 158, 158),
    (96, 125, 139)
)

BACKBONE = "resnet50"
FPN_CHANNELS = 256
NUM_MASK = 32
IMG_SIZE = 550

TOP_K = 200
CONF_THRESHOLD = 0.05
NMS_THRESHOLD = 0.3
MAX_NUM_DETECTION = 100



NUM_CLASSES = dict({
    "coco": 81,
    "pascal": 21,
    "your_custom_dataset": 0,
    "packet_types": 4,
    "packets": 2
})

DETECT = dict({
    "coco":{
        "num_cls": NUM_CLASSES["coco"],
        "label_background": 0,
        "top_k": TOP_K,
        "conf_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "max_num_detection": MAX_NUM_DETECTION},
    "pascal":{
        "num_cls": NUM_CLASSES["pascal"],
        "label_background": 0,
        "top_k": TOP_K,
        "conf_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "max_num_detection": MAX_NUM_DETECTION},
    "packet_types":{
        "num_cls": NUM_CLASSES["packet_types"],
        "label_background": 0,
        "top_k": TOP_K,
        "conf_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "max_num_detection": MAX_NUM_DETECTION},
    "packets":{
        "num_cls": NUM_CLASSES["packets"],
        "label_background": 0,
        "top_k": TOP_K,
        "conf_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "max_num_detection": MAX_NUM_DETECTION},
    
})

ANCHOR = dict({
    "coco": {"img_size": IMG_SIZE,
             "feature_map_size": [69, 35, 18, 9, 5],
             "aspect_ratio": [1, 0.5, 2],
             "scale": [24, 48, 96, 192, 384]},

    "pascal": {"img_size": IMG_SIZE,
               "feature_map_size": [69, 35, 18, 9, 5],
               "aspect_ratio": [1, 0.5, 2],
               "scale": [24 * (4 / 3), 48 * (4 / 3), 96 * (4 / 3), 192 * (4 / 3), 384 * (4 / 3)]},

    "packet_types": {"img_size": IMG_SIZE,
                         "feature_map_size": [69, 35, 18, 9, 5],
                         "aspect_ratio": [1, 0.5, 2],
                         "scale": [24, 48, 96, 192, 384]},
    "packets": {"img_size": IMG_SIZE,
                         "feature_map_size": [69, 35, 18, 9, 5],
                         "aspect_ratio": [1, 0.5, 2],
                         "scale": [24, 48, 96, 192, 384]}

})

CLASS_NAMES = dict({
    "coco" : ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    "pascal" : ("aeroplane", "bicycle", "bird", "boat", "bottle",
                  "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "person", "pottedplant",
                  "sheep", "sofa", "train", "tvmonitor"),
    "packets":  ('packet', "_"),
    "packet_types": ('big_000', 'kld_042', 'kld_054', 'kld_058')     
})
