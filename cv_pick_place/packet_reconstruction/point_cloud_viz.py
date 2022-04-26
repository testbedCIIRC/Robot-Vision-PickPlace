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
    def __init__(self, path):
        color_raw = o3d.io.read_image(path+"/color_image_crop.jpg")
        depth_raw = o3d.io.read_image(path+"/depth_image_crop.png")
        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
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
        # Rot_z = [[math.cos(math.radians(-20)),-math.sin(math.radians(-20)), 0.0, 0.0],
        #         [math.sin(math.radians(-20)), math.cos(math.radians(-20)), 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0]]
        # source.transform(Rot_z)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(source)
        # vis.add_geometry(target)
        icp_iteration = 100

        for i in range(icp_iteration):
            # source.transform(Rot_z)
            vis.update_geometry(source)
            vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()

# pclv = PointCloudViz("cv_pick_place/temp_rgbd")
# pclv.show_point_cloud()
