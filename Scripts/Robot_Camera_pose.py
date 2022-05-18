from opcua import Client
import datetime
import time
import os
import json
import cv2 
import math
from opcua import ua
import pyrealsense2 as rs
import numpy as np

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
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

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

circ_mov = {
"init_pos":[{'x':0.0,'y':-300.0,'z':270.0,'a':0.0,'b':0.0,'c':-131.0,'status':2,'turn':42},
            {'x':300.0,'y':0.0,'z':270.0,'a':90.0,'b':0.0,'c':-132.0,'status':2,'turn':34},
            {'x':254.6,'y':-254.6,'z':270.0,'a':45.0,'b':0.0,'c':-125.5,'status':2,'turn':42},
            {'x':254.6,'y':254.6,'z':270.0,'a':135.0,'b':0.0,'c':-125.5,'status':2,'turn':35}],

"aux_pos": [{'x':0.0,'y':0.0,'z':360.0,'a':0.0,'b':0.0,'c':-180.0,'status':2,'turn':35},
            {'x':0.0,'y':0.0,'z':360.0,'a':90.0,'b':0.0,'c':-180.0,'status':2,'turn':34},
            {'x':0.0,'y':0.0,'z':360.0,'a':45.0,'b':0.0,'c':180.0,'status':2,'turn':42},
            {'x':0.0,'y':0.0,'z':360.0,'a':135.0,'b':0.0,'c':180.0,'status':2,'turn':42}],

"end_pos": [{'x':0.0,'y':300,'z':270.0,'a':0.0,'b':0.0,'c':131.0,'status':2,'turn':35},
            {'x':-100.0,'y':0.0,'z':326.0,'a':90.0,'b':0.0,'c':161.0,'status':2,'turn':42},
            {'x':-254.6,'y':254.6,'z':270.0,'a':45.0,'b':0.0,'c':125.5,'status':2,'turn':35},
            {'x':-254.6,'y':-254.6,'z':270.0,'a':135.0,'b':0.0,'c':125.5,'status':2,'turn':10}]
          }
out_dict =  {"frames":[]}

if __name__ == '__main__':
    password = input('Enter password: ')
    client = Client("opc.tcp://user:"+str(password)+"@10.35.91.101:4840/")
    client.connect()
    print("client connected")
    Start_Prog = client.get_node('ns=3;s="HMIKuka"."robot"."example"."circle"."command"."start"')
    Rob_Stopped = client.get_node('ns=3;s="InstKukaControl"."instAutomaticExternal"."ROB_STOPPED"')
    Act_Pos_X = client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."X"')
    Act_Pos_Y = client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Y"')
    Act_Pos_Z = client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Z"')
    Act_Pos_A = client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."A"')
    Act_Pos_B = client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."B"')
    Act_Pos_C = client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."C"')
    Act_Pos_Turn = client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Turn"')
    Act_Pos_Status = client.get_node('ns=3;s="InstKukaControl"."instReadActualPos"."Status"')

    init_Circ_Pos_X = client.get_node('ns=3;s="InstCircle"."positions"[1]."E6POS"."X"')
    init_Circ_Pos_Y = client.get_node('ns=3;s="InstCircle"."positions"[1]."E6POS"."Y"')
    init_Circ_Pos_Z = client.get_node('ns=3;s="InstCircle"."positions"[1]."E6POS"."Z"')
    init_Circ_Pos_A = client.get_node('ns=3;s="InstCircle"."positions"[1]."E6POS"."A"')
    init_Circ_Pos_B = client.get_node('ns=3;s="InstCircle"."positions"[1]."E6POS"."B"')
    init_Circ_Pos_C = client.get_node('ns=3;s="InstCircle"."positions"[1]."E6POS"."C"')
    init_Circ_Pos_Status = client.get_node('ns=3;s="InstCircle"."positions"[1]."E6POS"."Status"')
    init_Circ_Pos_Turn = client.get_node('ns=3;s="InstCircle"."positions"[1]."E6POS"."Turn"')

    aux_Circ_Pos_X = client.get_node('ns=3;s="InstCircle"."positions"[2]."E6POS"."X"')
    aux_Circ_Pos_Y = client.get_node('ns=3;s="InstCircle"."positions"[2]."E6POS"."Y"')
    aux_Circ_Pos_Z = client.get_node('ns=3;s="InstCircle"."positions"[2]."E6POS"."Z"')
    aux_Circ_Pos_A = client.get_node('ns=3;s="InstCircle"."positions"[2]."E6POS"."A"')
    aux_Circ_Pos_B = client.get_node('ns=3;s="InstCircle"."positions"[2]."E6POS"."B"')
    aux_Circ_Pos_C = client.get_node('ns=3;s="InstCircle"."positions"[2]."E6POS"."C"')
    aux_Circ_Pos_Status = client.get_node('ns=3;s="InstCircle"."positions"[2]."E6POS"."Status"')
    aux_Circ_Pos_Turn = client.get_node('ns=3;s="InstCircle"."positions"[2]."E6POS"."Turn"')
    
    end_Circ_Pos_X = client.get_node('ns=3;s="InstCircle"."positions"[3]."E6POS"."X"')
    end_Circ_Pos_Y = client.get_node('ns=3;s="InstCircle"."positions"[3]."E6POS"."Y"')
    end_Circ_Pos_Z = client.get_node('ns=3;s="InstCircle"."positions"[3]."E6POS"."Z"')
    end_Circ_Pos_A = client.get_node('ns=3;s="InstCircle"."positions"[3]."E6POS"."A"')
    end_Circ_Pos_B = client.get_node('ns=3;s="InstCircle"."positions"[3]."E6POS"."B"')
    end_Circ_Pos_C = client.get_node('ns=3;s="InstCircle"."positions"[3]."E6POS"."C"')
    end_Circ_Pos_Status = client.get_node('ns=3;s="InstCircle"."positions"[3]."E6POS"."Status"')
    end_Circ_Pos_Turn = client.get_node('ns=3;s="InstCircle"."positions"[3]."E6POS"."Turn"')

    def change(ind):
        init_Circ_Pos_X.set_value(ua.DataValue(ua.Variant(circ_mov['init_pos'][ind]['x'], ua.VariantType.Float)))
        init_Circ_Pos_Y.set_value(ua.DataValue(ua.Variant(circ_mov['init_pos'][ind]['y'], ua.VariantType.Float)))
        init_Circ_Pos_Z.set_value(ua.DataValue(ua.Variant(circ_mov['init_pos'][ind]['z'], ua.VariantType.Float)))
        init_Circ_Pos_A.set_value(ua.DataValue(ua.Variant(circ_mov['init_pos'][ind]['a'], ua.VariantType.Float)))
        init_Circ_Pos_B.set_value(ua.DataValue(ua.Variant(circ_mov['init_pos'][ind]['b'], ua.VariantType.Float)))
        init_Circ_Pos_C.set_value(ua.DataValue(ua.Variant(circ_mov['init_pos'][ind]['c'], ua.VariantType.Float)))
        init_Circ_Pos_Status.set_value(ua.DataValue(ua.Variant(circ_mov['init_pos'][ind]['status'], ua.VariantType.Int16)))
        init_Circ_Pos_Turn.set_value(ua.DataValue(ua.Variant(circ_mov['init_pos'][ind]['turn'], ua.VariantType.Int16)))

        aux_Circ_Pos_X.set_value(ua.DataValue(ua.Variant(circ_mov['aux_pos'][ind]['x'], ua.VariantType.Float)))
        aux_Circ_Pos_Y.set_value(ua.DataValue(ua.Variant(circ_mov['aux_pos'][ind]['y'], ua.VariantType.Float)))
        aux_Circ_Pos_Z.set_value(ua.DataValue(ua.Variant(circ_mov['aux_pos'][ind]['z'], ua.VariantType.Float)))
        aux_Circ_Pos_A.set_value(ua.DataValue(ua.Variant(circ_mov['aux_pos'][ind]['a'], ua.VariantType.Float)))
        aux_Circ_Pos_B.set_value(ua.DataValue(ua.Variant(circ_mov['aux_pos'][ind]['b'], ua.VariantType.Float)))
        aux_Circ_Pos_C.set_value(ua.DataValue(ua.Variant(circ_mov['aux_pos'][ind]['c'], ua.VariantType.Float)))
        aux_Circ_Pos_Status.set_value(ua.DataValue(ua.Variant(circ_mov['aux_pos'][ind]['status'], ua.VariantType.Int16)))
        aux_Circ_Pos_Turn.set_value(ua.DataValue(ua.Variant(circ_mov['aux_pos'][ind]['turn'], ua.VariantType.Int16)))

        end_Circ_Pos_X.set_value(ua.DataValue(ua.Variant(circ_mov['end_pos'][ind]['x'], ua.VariantType.Float)))
        end_Circ_Pos_Y.set_value(ua.DataValue(ua.Variant(circ_mov['end_pos'][ind]['y'], ua.VariantType.Float)))
        end_Circ_Pos_Z.set_value(ua.DataValue(ua.Variant(circ_mov['end_pos'][ind]['z'], ua.VariantType.Float)))
        end_Circ_Pos_A.set_value(ua.DataValue(ua.Variant(circ_mov['end_pos'][ind]['a'], ua.VariantType.Float)))
        end_Circ_Pos_B.set_value(ua.DataValue(ua.Variant(circ_mov['end_pos'][ind]['b'], ua.VariantType.Float)))
        end_Circ_Pos_C.set_value(ua.DataValue(ua.Variant(circ_mov['end_pos'][ind]['c'], ua.VariantType.Float)))
        end_Circ_Pos_Status.set_value(ua.DataValue(ua.Variant(circ_mov['end_pos'][ind]['status'], ua.VariantType.Int16)))
        end_Circ_Pos_Turn.set_value(ua.DataValue(ua.Variant(circ_mov['end_pos'][ind]['turn'], ua.VariantType.Int16)))
        time.sleep(0.5)
    frame_num = -1
    bpressed = 0
    record = False
    dc = DepthCamera()
    with open('robot_camera_pose.json','w') as file:
        while True:
            rob_stopped = Rob_Stopped.get_value()
            start = Start_Prog.get_value()
            x_pos = Act_Pos_X.get_value()
            y_pos = Act_Pos_Y.get_value()
            z_pos = Act_Pos_Z.get_value()
            a_pos = Act_Pos_A.get_value()
            b_pos = Act_Pos_B.get_value()
            c_pos = Act_Pos_C.get_value()
            status_pos = Act_Pos_Status.get_value()
            turn_pos = Act_Pos_Turn.get_value()

            # print(start, x_pos, y_pos, z_pos, a_pos, b_pos, c_pos)

            Rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(math.radians(c_pos)),-math.sin(math.radians(c_pos))],
            [0.0, math.sin(math.radians(c_pos)), math.cos(math.radians(c_pos))] 
                            ])

            Rot_y = np.array([
            [math.cos(math.radians(b_pos)), 0.0, math.sin(math.radians(b_pos))],
            [0.0, 1.0, 0.0],
            [-math.sin(math.radians(b_pos)), 0.0, math.cos(math.radians(b_pos))]
                            ])

            Rot_z = np.array([
            [math.cos(math.radians(a_pos)),-math.sin(math.radians(a_pos)), 0.0],
            [math.sin(math.radians(a_pos)), math.cos(math.radians(a_pos)), 0.0],
            [0.0, 0.0, 1.0]
                            ])

            Rot_mat = Rot_x @ Rot_y @ Rot_z

            Trans_mat = np.array([
            [None, None, None, x_pos * 0.001],
            [None, None, None, y_pos * 0.001],
            [None, None, None, z_pos * 0.001],
            [0.0, 0.0, 0.0, 1.0]
                            ])
            Trans_mat[:3,:3] = Rot_mat[:,:]         

            ret, depth_frame, color_frame, colorized_depth = dc.get_frame()
            height, width, depth = color_frame.shape[0],color_frame.shape[1],color_frame.shape[2]
            color_frame = cv2.convertScaleAbs(color_frame, alpha=1.2, beta=10)

            if record:
                frame_num += 1
                print("frame: ",frame_num)           
                img_path = "./images"+"/"+str(frame_num)
                path_mat_dict = {"file_path":img_path,"transform_matrix":Trans_mat.tolist()}
                out_dict['frames'].append(path_mat_dict)
                img_path = os.path.join("images",str(frame_num))
                cv2.imwrite(img_path + ".jpg", color_frame)

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

            cv2.imshow("Frame", color_frame)

            key = cv2.waitKey(1)

            if key == 27:
                json.dump(out_dict, file)
                client.disconnect()
                cv2.destroyAllWindows()
                break

            if key == ord('r'):
                if record == False:
                    record = True
                else:
                    record = False

            if rob_stopped:
                if key == ord('b'):
                    bpressed += 1
                    if bpressed == 5:
                        Start_Prog.set_value(ua.DataValue(True))
                        print('Program Started: ',start)
                        time.sleep(0.7)
                        Start_Prog.set_value(ua.DataValue(False))
                        time.sleep(0.5)
                        bpressed = 0
                elif key != ord('b'):
                    bpressed = 0
            
            if key == ord('v'):
                ind = 0
                change(ind)
                print('changed to',ind)
            if key == ord('c'):
                ind = 1
                change(ind)
                print('changed to',ind)
            if key == ord('x'):
                ind = 2
                change(ind)
                print('changed to',ind)
            if key == ord('z'):
                ind = 3
                change(ind)
                print('changed to',ind)
    