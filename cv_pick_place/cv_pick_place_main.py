import os
from robot_communication.robot_control import robot_control

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