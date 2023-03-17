from robot_cell.detection.realsense_depth import DepthCamera
from robot_cell.detection.apriltag_detection import ProcessingApriltag

import matplotlib.pyplot as plt
import csv
import json
import numpy as np
import cv2
import scipy.linalg

CM2MM = 10

def transform_with_homography(homography_matrix, point_px):
    point_cm = np.matmul(homography_matrix, np.array([point_px[0], point_px[1], 1]))
    point_mm = (point_cm[0] * CM2MM, point_cm[1] * CM2MM)
    return point_mm


# Inititalize Apriltag Detector
apriltag = ProcessingApriltag()
apriltag.load_world_points("config/conveyor_points.json")

# camera = DepthCamera(config_path="config/D435_camera_config.json")
camera = DepthCamera()

# Get frames from camera
frame_captured = False
rgb_frame = None
i = 0
while (not frame_captured) or (rgb_frame is None) or i < 20:
    frame_captured, _, _, rgb_frame, _ = camera.get_frames()
    i += 1

frame_height, frame_width, frame_channel_count = rgb_frame.shape

apriltag.detect_tags(rgb_frame)
homography = apriltag.compute_homog()

x_range = range(1, frame_width - 2, 100)
y_range = range(200, frame_height - 200, 100)

num_x = 0
num_y = 0
for x_coord in x_range:
    num_x += 1
for y_coord in y_range:
    num_y += 1


x_frame_homography = np.ones((num_x, num_y))
y_frame_homography = np.ones((num_x, num_y))

x_frame_camera = np.ones((num_x, num_y))
y_frame_camera = np.ones((num_x, num_y))

fit_data = np.zeros((num_x * num_y, 3))

# 192, 68
x_idx = 0
y_idx = 0
for x_coord in x_range:
    for y_coord in y_range:
        x_homog, y_homog = transform_with_homography(homography, (x_coord, y_coord))
        x_frame_homography[x_idx, y_idx] = x_homog
        y_frame_homography[x_idx, y_idx] = y_homog
        x_camera, y_camera, _ = camera.pixel_to_3d_conveyor_frame((x_coord, y_coord))
        y_err = -0.00155943 * x_coord - 0.02827615 * y_coord + 0.92666205
        # y_err = 3.06703596e-02 + 5.21174391e-03 * x_coord + 9.66156093e-03 * y_coord - 6.01595281e-06 * x_coord * y_coord - 2.10701093e-06 * x_coord**2 - 8.60501561e-06 * y_coord**2
        y_camera = y_camera + y_err
        x_frame_camera[x_idx, y_idx] = x_camera
        y_frame_camera[x_idx, y_idx] = y_camera
        fit_data[(x_idx + 1) * y_idx, 0] = x_coord
        fit_data[(x_idx + 1) * y_idx, 1] = y_coord
        fit_data[(x_idx + 1) * y_idx, 2] = y_homog - y_camera
        y_idx += 1
    x_idx += 1
    y_idx = 0

# Plane fit
A = np.c_[fit_data[:, 0], fit_data[:, 1], np.ones(fit_data.shape[0])]
C, _, _, _ = scipy.linalg.lstsq(A, fit_data[:, 2])
print("Plane fit coefficients:", C)

# Curved plane fit
# A = np.c_[np.ones(fit_data.shape[0]), fit_data[:, :2], np.prod(fit_data[:, :2], axis=1), fit_data[:, :2]**2]
# C, _, _, _ = scipy.linalg.lstsq(A, fit_data[:, 2])
# print("Plane fit coefficients:", C)

# Plot
fig = plt.figure()
ax11 = fig.add_subplot(231, projection="3d")
ax12 = fig.add_subplot(232, projection="3d")
ax13 = fig.add_subplot(233, projection="3d")
ax21 = fig.add_subplot(234, projection="3d")
ax22 = fig.add_subplot(235, projection="3d")
ax23 = fig.add_subplot(236, projection="3d")
_x = np.arange(num_x)
_y = np.arange(num_y)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

# fitted_diff = C[0] * _xx + C[1] * _yy + C[2]

top = x_frame_homography.ravel("C")
bottom = np.zeros_like(top)
width = depth = 1
ax11.bar3d(x, y, bottom, width, depth, top, shade=True)
ax11.set_title("X Homography")

top = y_frame_homography.ravel("F")
bottom = np.zeros_like(top)
width = depth = 1
ax21.bar3d(x, y, bottom, width, depth, top, shade=True)
ax21.set_title("Y Homography")

top = x_frame_camera.ravel("C")
bottom = np.zeros_like(top)
width = depth = 1
ax12.bar3d(x, y, bottom, width, depth, top, shade=True)
ax12.set_title("X Camera")

top = y_frame_camera.ravel("F")
bottom = np.zeros_like(top)
width = depth = 1
ax22.bar3d(x, y, bottom, width, depth, top, shade=True)
ax22.set_title("Y Camera")

top = (x_frame_homography - x_frame_camera).ravel("C")
bottom = np.zeros_like(top)
width = depth = 1
ax13.bar3d(x, y, bottom, width, depth, top, shade=True)
ax13.set_title("X Difference")

top = (y_frame_camera - y_frame_homography).ravel("F")
bottom = np.zeros_like(top)
width = depth = 1
ax23.bar3d(x, y, bottom, width, depth, top, shade=True)
ax23.set_title("Y Difference")

plt.show(block=True)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(fit_data[:, 0], fit_data[:, 1], fit_data[:, 2], c='r', s=50)
# ax.plot_surface(_xx, _yy, fitted_diff, rstride=1, cstride=1)
# ax.axis('auto')
# ax.axis('tight')
# plt.show(block=True)


# x_px = []
# x_mm_homography = []
# x_mm_camera = []

# y_coord = frame_height // 2
# for x_coord in range(1, frame_width - 1):
#     x_px.append(x_coord)
#     x_mm_homography.append(transform_with_homography(homography, (x_coord, y_coord))[0])
#     x_mm_camera.append(camera.pixel_to_3d_conveyor_frame((x_coord, y_coord))[0])

# y_px = []
# y_mm_homography = []
# y_mm_camera = []

# x_coord = frame_width // 2
# for y_coord in range(1, frame_height - 1):
#     y_px.append(y_coord)
#     y_mm_homography.append(transform_with_homography(homography, (x_coord, y_coord))[0])
#     y_mm_camera.append(camera.pixel_to_3d_conveyor_frame((x_coord, y_coord))[0])

# plt.figure(0)
# plt.title("Changing X")
# plt.plot(x_mm_homography, label="Homography")
# plt.plot(x_mm_camera, label="Camera")
# plt.xlabel("mm")
# plt.ylabel("mm")
# plt.grid(True)
# plt.show(block=False)

# plt.figure(1)
# plt.title("Changing Y")
# plt.plot(y_mm_homography, label="Homography")
# plt.plot(y_mm_camera, label="Camera")
# plt.xlabel("mm")
# plt.ylabel("mm")
# plt.grid(True)
# plt.show(block=False)

# plt.figure(2)
# plt.title("X Error")
# plt.plot(np.array(x_mm_homography) - np.array(x_mm_camera), label="diff")
# plt.xlabel("mm")
# plt.ylabel("mm")
# plt.grid(True)
# plt.show(block=False)

# plt.figure(3)
# plt.title("Y Error")
# plt.plot(np.array(y_mm_homography) - np.array(y_mm_camera), label="diff")
# plt.xlabel("mm")
# plt.ylabel("mm")
# plt.grid(True)
# plt.show(block=True)
