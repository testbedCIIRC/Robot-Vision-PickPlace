import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import time
import copy
import math

class PointCloudViz():
    def __init__(self, path, packet, dims = (240,240)):

        self.path = path
        self.packet = packet
        self.dims = dims
        self.color_raw_path = "/color_image_crop.jpg"
        self.depth_raw_path = "/depth_image_crop.png"
        self.compute_depth_mean()
        color_raw = o3d.io.read_image(self.path + self.color_raw_path)
        depth_raw = o3d.io.read_image(self.path + self.depth_raw_path)
        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    
    def compute_depth_mean(self):
        cv2.namedWindow("Depth Frame Average")
        packet = self.packet
        depth_mean = np.mean(packet.depth_maps, axis=2)

        depth_frames_dim = depth_mean.shape
        print('depth_frames', depth_frames_dim)

        if 0 not in depth_frames_dim:
            depth_mean = cv2.resize(depth_mean, self.dims)
            clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 5))
            for i in range(0, int(packet.depth_maps.shape[2])):
                depth_frames = packet.depth_maps[:, :, i]
                depth_frames = cv2.resize(depth_frames, self.dims)

                depth_frame_hist = clahe.apply(depth_frames.astype(np.uint8))
                cv2_colorized_depth = cv2.applyColorMap(depth_frame_hist, cv2.COLORMAP_JET)
                cv2.imshow("Depth Frame Average", cv2.resize(cv2_colorized_depth.copy(), (680,680)))
                cv2.waitKey(50)
            depth_frame_hist = clahe.apply(depth_mean.astype(np.uint8))
            cv2_colorized_depth = cv2.applyColorMap(depth_frame_hist, cv2.COLORMAP_JET)
            cv2.imshow("Depth Frame Average", cv2.resize(cv2_colorized_depth.copy(), (680,680)))
            cv2.waitKey(1)

            cv2.imwrite(self.path + self.color_raw_path, cv2_colorized_depth)
            cv2.imwrite(self.path + self.depth_raw_path, depth_mean.astype(np.uint16))

    def create_point_cloud(self):
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            self.rgbd_image, o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        
        self.pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.005)
        self.pcd.estimate_normals()

    def show_point_cloud(self):
        self.create_point_cloud()
        source = copy.deepcopy(self.pcd)
        # target = self.pcd
        Rot_x_ang = -35
        Rot_x = [[1.0, 0.0, 0.0, 0.0],
                [0.0, math.cos(math.radians(Rot_x_ang)),-math.sin(math.radians(Rot_x_ang)),0.0],
                [0.0, math.sin(math.radians(Rot_x_ang)), math.cos(math.radians(Rot_x_ang)),0.0],
                [0.0, 0.0, 0.0, 1.0]]

        Rot_y_ang = 0     
        Rot_y = [[math.cos(math.radians(Rot_y_ang)), 0.0, math.sin(math.radians(Rot_y_ang)), 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-math.sin(math.radians(Rot_y_ang)), 0.0, math.cos(math.radians(Rot_y_ang)), 0.0],
                [0.0, 0.0, 0.0, 1.0]]

        Rot_z = [[math.cos(math.radians(-20)),-math.sin(math.radians(-20)), 0.0, 0.0],
                [math.sin(math.radians(-20)), math.cos(math.radians(-20)), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]
        source.transform(np.array(Rot_x)@np.array(Rot_y))

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(source)
        # vis.add_geometry(target)
        icp_iteration = 250

        for i in range(icp_iteration):
            # source.transform(Rot_z)
            vis.update_geometry(source)
            vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()

# pclv = PointCloudViz("cv_pick_place/temp_rgbd")
# pclv.show_point_cloud()
