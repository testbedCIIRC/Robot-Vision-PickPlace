from dis import dis
from tkinter import CENTER
from turtle import distance
from xml.dom.minidom import parseString
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy import signal
import math
import cv2

from robot_cell.packet.packet_object import Packet

M2MM = 1000.0
EPS = 0.000000001
CLASSES = {
    # TODO: FILL via detection YOLACT or just take somthing from previous things
    0: "big_white_packet",
    1: "small_white_packet",
    2: "medium_white_packet",
    3: "brown_packet",
    4: "catfood",
    5: "banana",
    6: "skittles",
    7: "ketchup",
    8: "toothpaste",
    9: "showergel",
    10: "mouthwash",
    11: "stain_remover",
    12: "trex",
}
# Just name of the classes IDK if they will be used
LINE_LIST = [1, 8]


class GripPositionEstimation:
    def __init__(
        self,
        visualize: bool = False,
        verbose: bool = False,
        center_switch: str = "mass",
        suction_cup_radius: float = 0.02,
        gripper_edge_length: float = 0.065,
        gripper_radius: float = 0.05,
        gripper_ration: float = 0.8,
        max_num_tries: int = 100,
        height_th: float = -0.76,
        num_bins: int = 20,
        black_list_radius: float = 0.01,
        save_depth_array: bool = False,
        mask_probability_ratio: float = 0.5,
    ):
        """
        Initializes class for predicting optimal position for the gripper.
        Every unit is in meters.

        Args:
            visualize (bool): If True will visualize the results.
            verbose (bool): If True will write some explanetions of whats happening.
            center_switch (str): "mass" or "height" - defines the center of the gripper.
            suction_cup_radius (float): Radius of the suction cup.
            gripper_edge_length (float): Length of the gripper(triangle) edge.
            gripper_radius (float): Radius of the gripper in meters.
            gripper_ration (float): Ratio of gripper radius for dettecting the gripper annulus.
            max_num_tries (int): Maximal number of tries to estimate the optimal position.
            height_th (float): Distance between camera and belt.
            num_bins (int): Number of bins for height thresholding (20 is good enough, 10 works as well).
            black_list_radius (float): Distance for blacklisting points.
            #TODO: Update docstring
        """

        self.visualization = visualize
        self.frame_num = 0

        if self.visualization:
            print(
                f"[GPE WARN]: Visualization while computing the pointcloud breaks the flow in real time"
            )
        self.verbose = verbose
        self.save_depth = save_depth_array
        self.gripper_edge_length = gripper_edge_length
        self.center_switch = center_switch
        self.suction_cup_radius = suction_cup_radius
        self.gripper_radius = gripper_radius
        self.gripper_ratio = gripper_ration  # Ration for computing gripper annulus
        self.blacklist_radius = black_list_radius
        self.simmilartiy_threshold = 0.9
        self.mask_probability_ratio = mask_probability_ratio

        # Pointcloud
        self.pcd = None
        self.points = None
        self.normals = None
        self.mask_threshold = None
        # Histograms
        self.th_idx = 0
        self.hist = None

        # Height threshold - defautly measured physically
        self.th_val = height_th
        self.num_bins = num_bins

        # Max number for tries for the estimation of best pose
        self.max_runs = max_num_tries
        self.run_number = 0  # current run
        self.spike_threshold = 0.04  # threshold for spike in the circlic neighborhood

    def _change_visualization(self, visualization: bool) -> None:
        """
        Changes the visualization state.
        Used for development.  # TODO: delete later

        Args:
            visualization (bool): New visualization state.
        """

        self.visualization = visualization

    def _create_pcd_from_rgb_depth_frames(
        self, rgb_name: str, depth_name: str, path: str = ""
    ) -> o3d.geometry.PointCloud:
        """
        Loads open3d PointCloud from rgb image and depth image stored in path into self.pcd.
        Used for development.

        Args:
            rgb_name (str): Name of the rgb image(.jpg).
            depth_name (str): Name of the depth image(.png).
            path (str): Path to the images.

        Returns:
            o3d.geometry.PointCloud: Pointcloud.
        """

        rgb = o3d.io.read_image(os.path.join(path, rgb_name))
        depth = o3d.io.read_image(os.path.join(path, depth_name))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            ),
        )
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd])

        return pcd

    def _load_numpy_depth_array_from_png(self, depth_name: str) -> np.ndarray:
        """
        Load png image and converts it to the numpy ndarray.
        Used for development.

        Args:
            depth_name (str): Path to the png file with depth.

        Returns:
            np.ndarray: Depth stored in 2D numpy ndarray.
        """

        image = Image.open(depth_name)
        return np.array(image)

    def _create_pcd_from_depth_array(
        self, depth_frame: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """
        Loads open3d PointCloud from depthframe.

        Args:
            depth_frame (np.ndarray): Depth values.

        Returns:
            o3d.geometry.PointCloud: Pointcloud.
        """

        depth = o3d.geometry.Image(np.ascontiguousarray(depth_frame).astype(np.uint16))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            ),
        )
        # o3d.visualization.draw_geometries([pcd])
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if self.visualization:
            o3d.visualization.draw_geometries([pcd])
        return pcd

    def _pcd_down_sample(self, voxel_size: float = 0.01) -> None:
        """
        Downsample PointCloud with given voxel size.

        Args:
            voxel_size (float): Voxel size for downsampling.
        """

        pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        self.pcd = pcd

    def _get_points_and_estimate_normals(self) -> tuple[np.ndarray]:
        """
        Estimates normals from PointCloud and reutrns both points coordinates and
        their respective normals in numpy arrays.

        Returns:
            tuple[np.ndarray, np.ndarray] : Vertex points and their normals.
        """

        self.pcd.estimate_normals()
        self.pcd_center = self.pcd.get_center()
        self.max_bound = self.pcd.get_max_bound()
        self.min_bound = self.pcd.get_min_bound()
        self.d_bound = self.max_bound - self.min_bound
        return np.asarray(self.pcd.points), np.asarray(self.pcd.normals)

    def _visualize_frame(self, points_dict: dict, title: str = f"Points 2D") -> None:
        """
        Visualize points in 2D, firstly all points secondly all sets points from
        points_dict, Uses key as legend and plots the numpy array values.

        Args:
            points_dict (dict): Dictionary with keys used as legend and values as numpy ndarrays.
            title (str): Title of the plot.
        """

        plt.figure(2)
        plt.scatter(self.points[:, 0], self.points[:, 1])
        legend = ("All points",)

        for key, vals in points_dict.items():
            if len(vals.shape) == 2:
                plt.scatter(vals[:, 0], vals[:, 1])
            else:
                plt.scatter(vals[0], vals[1])
            legend += (key,)

        plt.legend(legend)
        plt.title(title)
        plt.show()

    def _save_frame(
        self, points_dict: dict, title: str = f"Points 2D Projection"
    ) -> None:
        plt.figure(self.frame_num)

        plt.scatter(self.points[:, 0], self.points[:, 1])
        legend = ("All points",)

        for key, vals in points_dict.items():
            if len(vals.shape) == 2:
                plt.scatter(vals[:, 0], vals[:, 1])
            else:
                plt.scatter(vals[0], vals[1])
            legend += (key,)

        plt.legend(legend)
        plt.title(title)
        plt.savefig(f"{self.frame_num}.png", bbox_inches="tight")
        self.frame_num += 1

    def _visualize_histogram(self, n_bins: int) -> None:
        """
        Visualizes histogram with selected threshold.

        Args:
            n_bins (int): Number of bins for histogram.
        """

        fig, ax = plt.subplots()
        N, bins, patches = ax.hist(self.points[:, 2], bins=n_bins)
        for i in range(0, self.th_idx):
            patches[i].set_facecolor("#1f77b4")
        for i in range(self.th_idx, len(patches)):
            patches[i].set_facecolor("#ff7f0e")

        plt.title("Depth histogram")
        plt.xlabel("z [m]")
        plt.ylabel("count [-]")
        plt.show()

    def _hist_first_peak(self, th_count: int = 50) -> int:
        """
        Finds first peak in histogram.
        NOT SMART

        Args:
            th_count (int): Threshold value for finding the first peak with
                            at least 50 count.

        Returns:
            float: Height threshold between belt and object.
        """

        # assert len(self.hist.shape) == 1, "only 1D histogram"
        old = -np.inf
        s = 0
        for e, h in enumerate(self.hist):
            s += h
            if h < old and s > th_count:
                return e
            old = h

    def _histogram_peaks(self, num_total: int = 50) -> int:
        """
        Finds first peak in histogram.

        Args:
            th_count (int): Threshold value for finding the first peak with
                            atleast 50 count.

        Returns:
            int: Height threshold between belt and object.
        """

        peaks, _ = signal.find_peaks(self.hist[0])
        old = self.hist[0][peaks[0]]
        best_idx = peaks[0]
        s = 0
        count_th = 0.1 * num_total
        peak_found = False
        for peak in peaks:
            s = np.sum(self.hist[0][: peak + 1])
            if s < count_th:
                continue
            else:
                if not peak_found:
                    old = self.hist[0][peak]
                    peak_found = True

                if 1.0 + 0.2 > old / self.hist[0][peak] > 1.0 - 0.2:
                    best_idx = peak
                    old = self.hist[0][peak]
        return best_idx

    def _compute_histogram_threshold(
        self, values: np.ndarray, number_of_bins: int = 20
    ) -> None:
        """
        Finds height threshold via histogram stores it in itself.

        Args:
            values (np.ndarray): Values to be binned.
            number_of_bins (int): Number of bins to be used in histogram.
        """

        # assert len(values.shape) == 1, "Values should be 1D"
        h = np.histogram(values, bins=number_of_bins)
        self.hist = h
        # i = self._hist_first_peak()
        i = self._histogram_peaks(values.shape[0])
        self.th_idx = i + number_of_bins // 10 + 1
        # self.th_val = -0.76 Some constant used as dist from conv to camera
        self.th_val = h[1][self.th_idx]

    def _get_center_of_mass(self, points: np.ndarray) -> np.ndarray:
        """
        Get center of mass from points via approximation of mass with height (z axis).

        Args:
            points (np.ndarray): 3D points for computation.

        Returns:
            np.ndarray: Center of mass.
        """

        c_mass = np.average(
            points[:, :2], axis=0, weights=points[:, 2] - np.min(points[:, 2])
        )
        dists = np.linalg.norm(points[:, :2] - c_mass, axis=1)
        idx = np.argmin(dists)
        normal = self.normals[idx, :]
        c_mass = np.hstack((c_mass, self.points[idx, 2]))

        return c_mass, normal

    def _project_points2plane(
        self, center: np.ndarray, normal: np.ndarray, points: np.ndarray = None
    ) -> np.ndarray:
        """
        Projects points into the plane given by center and normal (unit normal).

        Args:
            center (np.ndarray): Center, origin of the plane, point on the plane.
            normal (np.ndarray): Normal of the plane.

        Returns:
            np.ndarray: Points projected into the plane.
        """
        if points is None:
            points = self.points
        dp = points - center
        dists = np.dot(dp, normal).reshape((-1, 1))
        projected = points - dists * normal
        return projected

    def _anuluss_mask(
        self,
        center: np.ndarray,
        normal: np.ndarray,
        points: np.ndarray = None,
        radius: float = None,
    ) -> np.ndarray:
        """
        Returns mask of pointes which are in annulus. Annulus is given by center
        and gripper radius.

        Args:
            center (np.ndarray): Center point of the annulus.
            normal (np.ndarray): Normal in the center point.
            points (np.ndarray): Points to be masked, If None, all points are used.
            radius (float): Radius of the annulus. If None, gripper radius is used.

        Returns:
            np.ndarray: Binary mask. True if point is in annulus.
        """
        if points is None:
            points = self.points
        if radius is None:
            radius = self.gripper_radius

        projected = self._project_points2plane(center, normal, points)
        l2 = np.linalg.norm(projected - center, axis=1)
        s = l2 >= radius * self.gripper_ratio
        b = l2 <= radius
        return b * s

    def _circle_mask(self, center: np.ndarray, radius: float = 0.05) -> np.ndarray:
        """
        Returns binary mask of points which are are closer to center than radius.

        Args:
            center (np.ndarray): Ceneter point.
            radius (float): Radius of the circle.

        Returns:
            np.ndarray: Binary mask.
        """
        l2 = np.linalg.norm(self.points - center, axis=1)
        return l2 <= radius

    def _fit_plane(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits plane to given points.

        Args:
            points (np.ndarray): Points to be fitted to plane.

        Returns:
            tuple[np.ndarray, np.ndarray]: Plane center and plane normal.
        """
        c = points.mean(axis=0)
        A = points - c
        M = np.dot(A.T, A)
        return c, np.linalg.svd(M)[0][:, -1]

    def _get_candidate_point(self, center: np.ndarray, allowed: np.ndarray):
        """
        Selects point closest to center, which is not blacklisted.

        Args:
            center (np.ndarray): Center of mass.
            allowed (np.ndarray): Binary mask of allowed points.

        Returns:
        """
        candidate_points = self.points[allowed, :]
        candidate_normals = self.normals[allowed, :]
        l2 = np.linalg.norm(self.points[allowed, :2] - center[:2], axis=1)
        idx = np.argmin(l2)
        candidate_point = candidate_points[idx, :]
        candidate_normal = candidate_normals[idx, :]
        return candidate_point, candidate_normal

    def _check_point_validity(self, point: np.ndarray, normal: np.ndarray) -> bool:
        """
        Checks if point is valid. No height spike in circlic neighborhood.
        Check if part of the outer circle is not on the conveyer belt.

        Args:
            point (np.ndarray): Point to be checked.
            normal (np.ndarray): Normal in the point for the plane of aproach.

        Returns:
            bool: True if point is valid, False otherwise.
        """

        # l2 (Euclidian) distance betwen point and all points projected into plane
        projected_points = self._project_points2plane(point, normal)
        l2 = np.linalg.norm(projected_points - point[:], axis=1)
        mask_inner = l2 < self.gripper_radius * self.gripper_ratio
        # If no points are in given l2 distatnce
        if not np.any(mask_inner):
            if self.verbose:
                print(
                    f"[GPE INFO]: No points in inner radius of gripper. Can not check the validity"
                )
            return False
        # Checks extremes in the  inside of the grapper
        inner_points = self.points[mask_inner, :]

        max_point = np.argmax(inner_points[:, 2])
        min_point = np.argmin(inner_points[:, 2])
        valid_max = np.abs(inner_points[max_point, 2] - point[2]) < self.spike_threshold
        valid_min = np.abs(inner_points[min_point, 2] - point[2]) < self.spike_threshold

        anls_mask = self._anuluss_mask(point, normal)
        if not np.any(anls_mask):
            if self.vervose:
                print(
                    f"[GPE INFO]: No points in anulus around the gripper. Can not check the validity."
                )
            return False

        # Checking if part of the gripper could get to the conv belt
        anls_points = self.points[anls_mask, :]
        lowest_point_idx = np.argmin(anls_points[:, 2])
        valid_conv = anls_points[lowest_point_idx, 2] > self.th_val

        validity = valid_max and valid_min and valid_conv
        if self.verbose:
            print(
                f"[GPE INFO]:Run: {self.run_number}. Point {point} is valid: {validity}"
            )
            if not valid_max:
                print(
                    f"\tReason - Spike:\tPoint {self.points[max_point, :]} difference in height is: {np.abs(self.points[max_point,2] - point[2])} spike threshold: { self.spike_threshold}"
                )
            if not valid_min:
                print(
                    f"\tReason - Spike:\tPoint {self.points[min_point, :]} difference in height is: {np.abs(self.points[min_point,2] - point[2])} spike threshold: {self.spike_threshold}"
                )
            if not valid_conv:
                print(
                    f"\tReason - Belt:\t Part of circle with point as center is on conveyer belt, i.e. point :{anls_points[lowest_point_idx, :]} Height threshold: {self.th_val}"
                )
        return validity

    def _expand_blacklist(self, point: np.ndarray) -> np.ndarray:
        """
        Expands blacklist by points which are closer than radius to the given
        point.

        Args:
            point (np.ndarray): Center point.
            radius (float): Threshold.

        Returns:
            np.ndarray: Updated binary mask for blacklist.
        """
        circle_mask = self._circle_mask(point, self.blacklist_radius)
        bl = np.logical_not(circle_mask)
        self.blacklist = np.logical_and(self.blacklist, bl)

    def _normal_similarity(self, normal_1: np.ndarray, normal_2: np.ndarray) -> float:
        """
        Checks whether the normals are relativelly the same.

        Args:
            normal_1 (np.ndarray): First normal.
            normal_2 (np.ndarray): Second normal.

        Returns:
            bool: True if normals are pointing in the same direction, False otherwise.
        """
        sim = np.dot(normal_1, normal_2)
        return sim

    def _point_pair_from_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        close_points: np.ndarray,
        close_normals: np.ndarray,
    ) -> tuple:
        """
        Recursivly selects best 2 points based on distance in close_points.
        Filters out points close to the edge and pairs with big normal difference measured by
        cosine simmilarity.

        Args:
            distance_matrix (np.ndarray): distance matrix between close points
            close_points (np.ndarray): close points from which we are locating the pick position
            close_normals (np.ndarray): _description_

        Returns:
            tuple: bool if pair is found, point_1, point_2, normal_1, normal_2 all np.ndarray
        """

        r, c = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        max_dist = distance_matrix[r, c]
        # set it to zero so it is removed from the matrix
        distance_matrix[r, c] = 0.0

        # Check neighrbood of each point if it is valid that is wheater they are far enough from the edge
        point_1, point_2 = close_points[r, :], close_points[c, :]
        neigh_1 = self.points[
            self._circle_mask(point_1, self.suction_cup_radius * self.real2pcd), :
        ]
        neigh_2 = self.points[
            self._circle_mask(point_2, self.suction_cup_radius * self.real2pcd), :
        ]
        val_p_1 = neigh_1[np.argmin(neigh_1[:, 2]), 2] > self.th_val
        val_p_2 = neigh_2[np.argmin(neigh_2[:, 2]), 2] > self.th_val

        # If point is not valid set all combinations with that point to zero
        if not val_p_1:
            distance_matrix[r, :] = 0.0
            distance_matrix[:, r] = 0.0
        if not val_p_2:
            distance_matrix[:, c] = 0.0
            distance_matrix[c, :] = 0.0

        # Check normals simmilarity, using cosine similarity
        normal_1, normal_2 = close_normals[r, :], close_normals[c, :]
        simmilarity = self._normal_similarity(normal_1, normal_2)
        simmilar = simmilarity > self.simmilartiy_threshold

        if self.verbose:
            print(
                f"[GPE INFO]: Points {point_1}, {point_2} are {val_p_1}, {val_p_2}\n\t\t Normals {normal_1}, {normal_2}\n\\t have similarity: {simmilarity}: {simmilar}"
            )
        if val_p_1 and val_p_2 and simmilar:
            return True, point_1, point_2, normal_1, normal_2
        if max_dist == 0.0:
            return False, None, None, None, None
        return self._point_pair_from_distance_matrix(
            distance_matrix, close_points, close_normals
        )

    def _pose_for_2points(
        self, center: np.ndarray, normal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find 2 points for gripping of the object.

        Args:
            center (np.ndarray): Center of mass.
            normal (np.ndarray): Normal in the point for the plane of aproach.
        Returns:
            tuple[np.ndarray,np.ndarray, np.ndarray]: point, normal and direction of y axis for gripping
        """
        # TODO: Write some visualization for to know what are you suppose to pick and where
        # Here is some problem with units

        # Now it should be working as intended
        real2pcd = self.d_bound[0] / (self.mm_width)
        direction_y = np.array([1.0, 0.0, 0.0])
        valid, simmilar, searched = False, False, False
        edge_length = self.gripper_edge_length * M2MM / 2 * real2pcd
        self.real2pcd = real2pcd

        while self.run_number < self.max_runs:
            # Find points and normals in annulus around center
            close_mask = self._anuluss_mask(
                center, normal, self.filtered_points, edge_length
            )
            # Add behaviour when close mask is empty, though it should not happen
            close_points = self.filtered_points[close_mask, :]
            close_normals = self.filtered_normals[close_mask, :]
            print(close_points.shape)
            # BLACKMAGIC NUMPY BROADCASTING
            distance_matrix = np.linalg.norm(
                close_points[:, None, :] - close_points, axis=2
            )
            # Take just one half (triangle) of the distance matrix
            distance_matrix = np.tril(distance_matrix)
            # Filter out combination with small distance Points that are colose to each other
            distance_matrix[distance_matrix < 0.7 * edge_length] = 0.0

            (
                valid,
                point_1,
                point_2,
                normal_1,
                normal_2,
            ) = self._point_pair_from_distance_matrix(
                distance_matrix, close_points, close_normals
            )

            if not valid:
                self.run_number += 1
                # compute new center and normal
                self._expand_blacklist(center)
                allowed_mask = np.logical_and(self.packet_mask, self.blacklist)
                searched = not np.any(allowed_mask)
                # All posible points were searched found no optimal point
                if searched:
                    print(
                        f"[GPE INFO]: All points were searched. No optimal point found."
                    )
                    break
                center, normal = self._get_candidate_point(center, allowed_mask)
                continue
            else:
                # NOTE: Maybe check if the points are colinear with center
                # Might not be necessary as the iteration trough maximum distance will probably find the colinear points
                direction_x = point_2 - point_1
                direction_x /= np.linalg.norm(direction_x)
                mid = (point_1 + point_2) / 2.0
                direction_z = (normal_1 + normal_2) / 2.0
                if direction_z[2] < 0.0:
                    if self.verbose:
                        print(f"[GPE INFO]: Inverted direction_z")
                    direction_z = -direction_z

                dir_y = np.cross(direction_z, direction_x)
                direction_y = dir_y / np.linalg.norm(dir_y)
                if direction_y[0] < 0:
                    direction_y = -direction_y
                c = (
                    direction_y
                    * self.real2pcd
                    * self.gripper_edge_length
                    * 1000
                    * math.sqrt(3)
                    / 6.0
                )
                center = mid + c
                third = (
                    mid
                    + direction_y
                    * self.real2pcd
                    * self.gripper_edge_length
                    * 1000
                    * math.sqrt(3)
                    / 2.0
                )

                if self.visualization:
                    viz_dict = {
                        "Height filtered points": self.filtered_points,
                        "Close points": close_points,
                        "pick_points": np.vstack((point_1, point_2)),
                        "center": center,
                    }
                    # self._visualize_frame(viz_dict)
                # TODO: Delete later
                viz_dict = {
                    "Height filtered points": self.filtered_points,
                    "Close points": close_points,
                    "pick_points": np.vstack((point_1, point_2)),
                    "m": mid,
                    "center": center,
                    "third": third,
                }
                self._save_frame(viz_dict)

                # print(normal, direction_z, normal - direction_z)
                # print(np.linalg.norm(normal), np.linalg.norm(direction_z), np.linalg.norm(normal - direction_z))
                print(
                    f"[DEL]: mid point:{mid}, \n\tax: {direction_x}, \n\tay: {direction_y}, \n\tax:{direction_z}"
                )
                return center, direction_z, direction_y

            print(f"[GPE INFO]: Run number {self.run_number} Did not find valid point")
            return None, None, None

    def _pose_for_circle(
        self, center: np.ndarray, normal: np.ndarray
    ) -> tuple[np.ndarray]:
        """
        compute the pose items that are large enough to contain the circle

        Args:
            center (np.ndarray): Original point of  the circel
            normal (np.ndarray): Normal of the original point

        Returns:
            tuple[np.ndarray]: Tuple
        """

        n_mask = self._anuluss_mask(center, normal)
        plane_c, plane_n = self._fit_plane(self.points[n_mask, :])

        valid = self._check_point_validity(center, plane_n)

        # this direction is basicaly set for this
        direction_y = np.array([-1.0, 0.0, 0.0])

        # Returns original point as optimal if it is valid
        if valid:
            if self.verbose:
                print(f"[INFO]: Picked original point as it is valid")
            if self.visualization:
                viz_dict = {
                    "Height filtered points": self.filtered_points,
                    "Neighborhood": self.points[n_mask, :],
                    "center point " + self.center_switch: center,
                    "center plane": plane_c,
                }
                self._visualize_frame(viz_dict)
            center[2] = plane_c[2]
            return center, plane_n, direction_y

        self._expand_blacklist(center)
        while self.run_number < self.max_runs:
            allowed_mask = np.logical_and(self.packet_mask, self.blacklist)
            searched = not np.any(allowed_mask)

            # All posible points were searched found no optimal point
            if searched:
                break

            c_point, c_normal = self._get_candidate_point(center, allowed_mask)
            valid = self._check_point_validity(c_point, c_normal)

            if not valid:
                self._expand_blacklist(c_point)
                self.run_number += 1
                continue

            n_mask = self._anuluss_mask(c_point, c_normal)
            neighbourhood_points = self.points[n_mask, :]
            plane_c, plane_n = self._fit_plane(neighbourhood_points)

            if self.visualization:
                viz_dict = {
                    "Height filtered points": self.filtered_points,
                    "Neighborhood": self.points[n_mask, :],
                    "center point " + self.center_switch: center,
                    "center plane": plane_c,
                }
                self._visualize_frame(viz_dict)
            c_point[2] = plane_c[2]
            return c_point, plane_n, direction_y

        if self.verbose:
            print(
                f"[GPE WARN]: Could not find the valid point, retrurning None. Reason: "
            )
            if searched:
                print(f"\tAll possible points were checked, did not found the optimal")
            else:
                print(f"\tExceded given number of tries: {self.max_runs}")
        return None, None, None

    def _detect_point_from_pcd(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect optimal point from pointcloud.
        FIXME: Update docstring
        Args:
            item (item): item class

        Returns:
            tuple[np.ndarray, np.ndarray]: Plane center and plane normal (x, y, z).
        """

        self._pcd_down_sample()
        self.points, self.normals = self._get_points_and_estimate_normals()
        if self.visualization:
            o3d.visualization.draw_geometries([self.pcd])
        if self.mask_threshold is None:
            print(f"[GPE INFO]: Mask not provided continue from depth histogeram")
            self._compute_histogram_threshold(self.points[:, 2], self.num_bins)

            if self.visualization:
                self._visualize_histogram(self.num_bins)
        else:
            # Converst the threshold value from mask into pcd units
            self.mask_threshold /= -M2MM
            # self.mask_threshold += + 0.0005
            self.th_val = self.mask_threshold

        # print(self.mask_threshold)
        # print(np.unique(self.points[:,2] + EPS, return_counts=True))
        self.packet_mask = self.points[:, 2] >= self.mask_threshold
        self.filtered_points = self.points[self.packet_mask, :]
        self.filtered_normals = self.normals[self.packet_mask, :]
        self.blacklist = np.full((self.points.shape[0],), True)

        # Compute original point
        if self.center_switch == "mass":
            center, normal = self._get_center_of_mass(self.filtered_points)
        elif self.center_switch == "height":
            idx = np.argmax(self.filtered_points[:, 2])
            center, normal = self.points[idx], self.normal[idx]

        if self.pick_type == "line":
            # Items that are too small for the whole gripper gripping by 2
            if self.verbose:
                print(
                    f"[GPE INFO]: Item {CLASSES[self.item_type]} in line list - Fitting 2 optimal points"
                )
            center_f, normal_z, direction_y = self._pose_for_2points(center, normal)
        else:
            # Items gripped by circle
            if self.verbose:
                print(
                    f"[GPE INFO]: Item {CLASSES[self.item_type]} in circle list -  Fitting circle"
                )
            # Everything else
            center_f, normal_z, direction_y = self._pose_for_circle(center, normal)

        return center_f, normal_z, direction_y

    def _get_relative_coordinates(
        self, point: np.ndarray, anchor: np.ndarray
    ) -> np.ndarray:
        """
        Returns relative ratio coordinates of point from center of gripper.

        Args:
            point (np.ndarray): Point to be transformed.
            anchor (np.ndarray): Relative coordinates of anchor point(center(px, py)).

        Returns:
            np.ndarray: Relative coordinates of point to anchor and height(px, py, z)
                        in packet coord system.
        """
        point_rel = (point[:2] - self.min_bound[:2]) / self.d_bound[:2]
        point_anchor_rel = point_rel - anchor

        return np.hstack((point_anchor_rel, point[2]))

    # XXX: HERE some eroor while recomputing with the line
    def _convert2actual_position(
        self,
        center: np.ndarray,
        anchor: np.ndarray,
        normal: np.ndarray,
        direction: np.ndarray,
        z_lim: tuple[float],
        rot_offset: float = -10.0,
        rotation_matrix: np.ndarray = np.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    ) -> np.ndarray:
        """
        Computes actual position for the robot. That is x, y, z, a(roll), b(pitch), c(yaw).

        Args:
            center (np.ndarray): Center of the gripper in pcd coord system.
            anchor (np.ndarray): Anchor of the gripper
            normal (np.ndarray): Normal of the gripper in pcd coord system.
            direction (np.ndarray): Direction of the gripper in pcd coord system.
            z_lim (tuple[float]): z limits of the gripper.
            rot_offset (float): y offset in degrees for the straight line
            rotation_matrix (np.ndarray): Rotation matrix between pcd and actual coord system.
        Returns:
            np.ndarray: Actual position for the robot.
        """
        z_min, z_max = z_lim

        new_z = -1 * (rotation_matrix @ normal)
        direction = rotation_matrix @ direction
        base_coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        if self.verbose:
            print(f"[GPE INFO]: Aproach vector {new_z} in coord base")

        if self.pick_type == "line":
            # Pick by 2 points
            new_y = direction
            new_x = np.cross(new_y, new_z)
            if new_y[0] < 0.0:
                new_x *= -1.0
                new_y *= -1.0
            print(f"{new_x}, \n {new_y}, \n {normal}")
            point_relative = self._get_relative_coordinates(center, anchor)
            dx, dy, z = point_relative
            shift_x, shift_y = -1 * dx * self.mm_width, -1 * dy * self.mm_height

            # XXX: Recalculation of the coords
            # dist = self.gripper_edge_length * math.sqrt(3) / 6.0
            # point_relative += np.array(dist * new_y)
            pack_z = abs(-1.0 * M2MM * z + self.th_val * M2MM) - 5.0
            pack_z = np.clip(pack_z, z_min, z_max)
            # coords = np.array([shift_x, shift_y, pack_z]) - dist * new_y
            coords = np.array([shift_x, shift_y, pack_z])

        else:
            point_relative = self._get_relative_coordinates(center, anchor)
            dx, dy, z = point_relative
            shift_x, shift_y = -1 * dx * self.mm_width, -1 * dy * self.mm_height
            # Changes the z value to be positive ,converts m to mm and shifts by the conv2cam_dist
            pack_z = abs(-1.0 * M2MM * z + self.th_val * M2MM) - 5.0
            pack_z = np.clip(pack_z, z_min, z_max)
            roll, pitch, yaw = self._vectors2RPYrot(normal, direction=direction)
            point_relative[0] += anchor[0]
            point_relative[1] += anchor[1]

            new_y = direction
            new_y[2] = -(np.dot(new_z[:2], new_y[:2])) / new_z[2]
            new_y /= np.linalg.norm(new_y)
            new_x = np.cross(new_y, new_z)
            coords = np.array([shift_x, shift_y, pack_z])

        # Normal with direction to angle
        alpha = np.arctan2(
            np.linalg.norm(np.cross(base_coords[:2, 0], new_x[:2])),
            np.dot(base_coords[:2, 0], new_x[:2]),
        )
        Rz = np.array(
            [
                [math.cos(alpha), -math.sin(alpha), 0],
                [math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 1],
            ]
        )
        coords_z = base_coords @ Rz

        # Now to rotate to the angle for x' ->x_new
        beta = np.arctan2(
            np.linalg.norm(np.cross(coords_z[:, 0], new_x)),
            np.dot(coords_z[:, 0], new_x),
        )
        beta *= -1 if new_x[2] >= 0 else 1
        Ry = np.array(
            [
                [math.cos(beta), 0, math.sin(beta)],
                [0, 1, 0],
                [-math.sin(beta), 0, math.cos(beta)],
            ]
        )
        coords_y = coords_z @ Ry

        # Now rotation from z''-> z_new
        gamma = np.arctan2(
            np.linalg.norm(np.cross(coords_y[:, 2], new_z)),
            np.dot(coords_y[:, 2], new_z),
        )
        gamma *= 1 if new_y[2] >= 0 else -1
        # This could be redundant Just for control if calculated and wanted bases are correct
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(gamma), -math.sin(gamma)],
                [0, math.sin(gamma), math.cos(gamma)],
            ]
        )
        # new_coords = coords_y @ Rx

        # For debugging purposes
        # if self.verbose:
        #     print(f"[INFO]: transformed base by the angles:\
        #             \n x: {new_coords[:,0]}\n y: {new_coords[:,1]}\n z: {new_coords[:,2]}")
        #     print(f"[INFO]: Delta between bases(Should be ZERO)\
        #             \n x: {new_x - new_coords[:,0]}\n y: {new_y - new_coords[:,1]}\n z: { new_z -new_coords[:,2]}")

        # Final array of angles in degrees
        angles = np.rad2deg(np.array([alpha, beta, gamma]))
        angles[0] += rot_offset
        return coords, angles

    # Remove later
    def _vectors2RPYrot(
        self,
        vector: np.ndarray,
        rotation_matrix: np.ndarray = np.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        direction=np.array([1.0, 0.0, 0.0]),
    ) -> np.ndarray:
        """
        Compute roll, pitch, yaw angles based on the vector, which is firstly
        transformed into the actual position coordinates by the rotation matrix.

        Args:
            vector (np.ndarray): Unit vector of the wanted position.
            rotation_matrix (np.ndarray): Rotation matrix between the packet base and coord base
                                          Default is that packet and base coords systems are
                                          rotated 180 degrees around z axis.
            direction (np.ndarray): Direction of one of the new axis (currently set [1, 0, ?]).

        Returns:
            np.ndarray: Angles (roll, pitch, yaw) in degrees.
        """
        # NOTE: This function is probably not used and obsolete
        # Can be removed in the future
        # Transformation of the vector into base coordinates
        new_z = -1 * (rotation_matrix @ vector)
        if self.verbose:
            print(f"[GPE INFO]: Aproach vector {new_z} in coord base")

        # base of the coordinate system used for recalculation of the angles
        base_coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        new_y = direction
        # NOTE: Computation of new base base based around the given z axis
        #   Selected y axis to be in the direction of the conv belt (ie 1, 0, ?)
        #   Recalculation of the lacs element so it lies in the plane
        #   x axis is then calculated to give orthogonal and follow right hand rule
        new_y[2] = -(np.dot(new_z[:2], new_y[:2])) / new_z[2]
        new_y /= np.linalg.norm(new_y)
        new_x = np.cross(new_y, new_z)

        # For debugging purposes
        # if self.verbose:
        #     print(f"[INFO]: New base for picking\
        #             \n x: {new_x}, norm: {np.linalg.norm(new_x):.2f}\
        #             \n y: {new_y}, norm: {np.linalg.norm(new_y):.2f}\
        #             \n z: {new_z}, norm: {np.linalg.norm(new_z):.2f}")

        # Rotations between angles x -> projection of x_new to xy plane
        alpha = np.arctan2(
            np.linalg.norm(np.cross(base_coords[:2, 0], new_x[:2])),
            np.dot(base_coords[:2, 0], new_x[:2]),
        )
        Rz = np.array(
            [
                [math.cos(alpha), -math.sin(alpha), 0],
                [math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 1],
            ]
        )
        coords_z = base_coords @ Rz

        # Now to rotate to the angle for x' ->x_new
        beta = np.arctan2(
            np.linalg.norm(np.cross(coords_z[:, 0], new_x)),
            np.dot(coords_z[:, 0], new_x),
        )
        beta *= -1 if new_x[2] >= 0 else 1
        Ry = np.array(
            [
                [math.cos(beta), 0, math.sin(beta)],
                [0, 1, 0],
                [-math.sin(beta), 0, math.cos(beta)],
            ]
        )
        coords_y = coords_z @ Ry

        # Now rotation from z''-> z_new
        gamma = np.arctan2(
            np.linalg.norm(np.cross(coords_y[:, 2], new_z)),
            np.dot(coords_y[:, 2], new_z),
        )
        gamma *= 1 if new_y[2] >= 0 else -1
        # This could be redundant Just for control if calculated and wanted bases are correct
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(gamma), -math.sin(gamma)],
                [0, math.sin(gamma), math.cos(gamma)],
            ]
        )
        # new_coords = coords_y @ Rx

        # For debugging purposes
        # if self.verbose:
        #     print(f"[INFO]: transformed base by the angles:\
        #             \n x: {new_coords[:,0]}\n y: {new_coords[:,1]}\n z: {new_coords[:,2]}")
        #     print(f"[INFO]: Delta between bases(Should be ZERO)\
        #             \n x: {new_x - new_coords[:,0]}\n y: {new_y - new_coords[:,1]}\n z: { new_z -new_coords[:,2]}")

        # Final array of angles in degrees
        angles = np.rad2deg(np.array([alpha, beta, gamma]))
        return angles

    def _get_pick_type(self, item: Packet) -> None:
        """
        Determine picking type based on item characteristics.
        # TODO: Replace with some dicisions by the packet characteristics

        Args:
            item (Packet): Packet to be picked.
        """
        self.item_type = item.type
        print(self.item_type)
        self.pick_type = "line"  # if item.type in LINE_LIST else "circle"

    def estimate_from_images(
        self,
        rgb_image_name: str,
        depth_image_name: str,
        path: str = "",
        anchor: np.array = np.array([0.5, 0.5]),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        NOTE: Propably obsolete used for development.
        Estimates point and normal from given images.
        (The idea is to call this function and it returns everything  needed)

        Args:
            rgb_image_name (str): Name of rgb image.
            depth_image_name (str): Name of depth image.
            path (str): Path to images.
            anchor (np.ndarray): Relative coordinates of anchor point(center(px, py)).

        Returns:
            tuple[np.ndarray, np.ndarray]: Point coordinates and its normal.
        """

        self.pcd = self._create_pcd_from_rgb_depth_frames(
            rgb_image_name, depth_image_name, path
        )
        center, normal = self._detect_point_from_pcd()

        if center is None:
            print(f"[GPE INFO]: Did not found optimal point")
            return center, normal
        relative = self._get_relative_coordinates(center, anchor)

        return relative, normal

    def estimate_from_depth_array(
        self, depth_array: np.ndarray, packet_mask: np.ndarray = None
    ) -> tuple[np.ndarray]:
        """
        Estimates optimal point from depth array.

        Args:
            depth_array (np.ndarray): Depth values.
            packet_mask (np.ndarray): Binary mask with packet, If None continue with all depths.
            anchor (np.ndarray): Relative coordinates of anchor point(center(px, py)).
            item (int): Item class idx, on which behaviour is

        Returns:
            tuple[np.ndarray]: Point and normal for picking.
        """
        # TODO: Later add to this stuff detection from packet
        # Now here due to the debugging purposes

        if packet_mask.shape != depth_array.shape:
            if self.verbose:
                print(
                    f"[GPE WARN]: Incompatible sizes: Binary mask {packet_mask.shape},Depth {depth_array.shape}"
                )
            return None, None

        # Sets everything outside of mask as lower
        if packet_mask is not None:
            belt = np.logical_not(packet_mask) * depth_array
            packet = packet_mask * depth_array
            # Selects lowest value as the threshold for the
            self.mask_threshold = max(
                np.min(belt[np.nonzero(belt)]), np.max(packet[np.nonzero(packet)])
            )
            print(
                f"[GPE INFO]: Selected depth threshold from mask: {self.mask_threshold}"
            )
            depth_array[np.logical_not(packet_mask)] = self.mask_threshold + 1

        # Creates PCD with threshold based on
        self.pcd = self._create_pcd_from_depth_array(depth_array)
        center, normal, direction = self._detect_point_from_pcd()

        # If nothing was found
        if center is None:
            return None, None, None

        # Make sure taht the normal vector has positive z
        if normal[2] < 0:
            print(f"[GPE INFO]: Inverted direction of the normal")
            normal *= -1

        return center, normal, direction

    def estimate_from_packet(
        self,
        packet: Packet,
        z_lim: tuple,
        y_lim: tuple,
        packet_coords: tuple,
        conv2cam_dist: float = 777.0,
        blacklist_radius: float = 0.01,
    ) -> tuple[float]:
        """
        Estimates optimal point of the packet.

        Args:
            packet (Packet): Packet for the estimator.
            z_lim (tuple): Limit for the robot gripper for not coliding (z_min, z_max).
            y_lim (tuple): Limit for the robot gripper conv belt.
            packet_coords (tuple): Cordinates of the packet.
            conv2cam_dist (float): Distance from conveyer to camera in mm.
            black_list_radius(float): Radius for blacklisting.

        Returns:
            tuple[float]: shift_in_x, shift_in_y, height_in_z,
                          roll, pitch, yaw
        """
        # TODO: add decision based on ration and sizes of packet bbox
        # MAYBE RENAME PACKET TO ITEM OR SOMETHING LIKE THAT
        # TODO: Add recalculation of coords by the real values
        # Creating new class with given params
        self.blacklist_radius = blacklist_radius
        # self._change_visualization(True)

        z_min, z_max = z_lim
        y_min, y_max = y_lim
        _, pack_y = packet_coords
        depth_frame = packet.avg_depth_crop

        self._get_pick_type(packet)

        pack_z = z_min
        shift_x, shift_y = None, None
        roll, pitch, yaw = None, None, None

        depth_exist = depth_frame is not None
        point_exists = False
        self.mm_width = packet.width_bnd_mm
        self.mm_height = packet.height_bnd_mm

        if depth_exist:
            mask = packet.mask
            # print(mask.shape, np.unique(mask, return_counts=True))
            mask = mask > 0
            # print(mask.shape, np.unique(mask, return_counts=True))
            if self.save_depth:
                # NOTE: JUST FOR SAVING THE TEST IMG, DELETE MAYBE
                print(f"[GPE INFO]: SAVING the depth array of packet {packet.id}")
                np.save(f"depth_array{packet.id}_precrop.npy", depth_frame)
                np.save(f"depth_array{packet.id}_precrop_mask.npy", mask)

            # Cropping of depth map and mask in case of packet being on the edge of the conveyor belt
            pack_y_max = pack_y + packet.height_bnd_mm / 2
            pack_y_min = pack_y - packet.height_bnd_mm / 2
            dims = depth_frame.shape
            # TODO: Add decision based on ratio and size of the bbox
            ratio = 0.5
            if pack_y_max > y_max:
                ratio_over = abs(pack_y_max - y_max) / packet.height_bnd_mm
                k = 1 - ratio_over
                self.mm_height *= k
                depth_frame = depth_frame[: int(k * dims[0]), :]
                ratio /= k
                mask = mask[: int(k * dims[0]), :]

            if pack_y_min < y_min:
                ratio_over = abs(pack_y_min - y_min) / packet.height_bnd_mm
                k = 1 - ratio_over
                self.mm_height *= k
                depth_frame = depth_frame[int(ratio_over * dims[0]) :, :]
                ratio = (0.5 - ratio_over) / k
                mask = mask[int(ratio_over * dims[0]) :, :]

            # Anchor for relative coordinates
            anchor = np.array([0.5, ratio])

            if self.save_depth:
                # NOTE: JUST FOR SAVING THE TEST IMG, DELETE MAYBE
                print(f"[GPE INFO]: SAVING the depth array of packet {packet.id}")
                np.save(f"depth_array{packet.id}_postcrop.npy", depth_frame)
                np.save(f"depth_array{packet.id}_postcrop_mask.npy", mask)

            # Estimates the point and normal
            center, normal, direction = self.estimate_from_depth_array(
                depth_frame, mask
            )
            point_exists = center is not None

            if point_exists:
                coord, angles = self._convert2actual_position(
                    center, anchor, normal, direction, z_lim
                )
                shift_x, shift_y, pack_z = coord
                roll, pitch, yaw = angles

        if self.verbose:
            print(f"[GPE INFO]: Optimal point found: {depth_exist and  point_exists}")
            if not depth_exist:
                print(f"\tReason - Average depth frame is None. Returns None")
            if not point_exists:
                print(f"\tReason - Valid point was not found. Returns None")
        # TODO: returm should not have coords as it is in cm not relative for pixels
        return shift_x, shift_y, pack_z, roll, pitch, yaw, coord
        # return shift_x, shift_y, pack_z, roll, pitch, yaw,


def main():
    # NOTE: Demo of how to work with this
    # Size of the triangle edge and radius of the circle
    triangle_edge = 0.085  # in meters
    gripper_radius = triangle_edge / np.sqrt(3)

    gpe = GripPositionEstimation(
        visualize=False,
        verbose=True,
        center_switch="mass",
        suction_cup_radius=0.02,
        gripper_edge_length=triangle_edge,
        gripper_radius=gripper_radius,
        gripper_ration=0.8,
    )
    gpe.pick_type = "line"
    # Toothpaste
    depth_array = np.load(
        os.path.join("robot_cell", "packet", "data", "new_items", "8_depth_array.npy")
    )
    img = cv2.imread(
        os.path.join("robot_cell", "packet", "data", "new_items", "8_rgb.jpg")
    )

    x_f = 784
    x_t = x_f + 355
    y_f = 490
    y_t = y_f + 140
    img = img[y_f:y_t, x_f:x_t, :]
    depth_array = depth_array[y_f:y_t, x_f:x_t]
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnt = tuple()
    mask = np.zeros_like(img)
    for i in range(3):
        imgray = img[:, :, i]
        ret, thresh = cv2.threshold(imgray, 35, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        p = img.copy()
        # cv2.drawContours(p, contours, -1, (0,255,0), 1)
        # cv2.imshow("img", p)
        # cv2.waitKey(0)
        mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    # cond = np.repeat((depth_array < 770)[:, :, np.newaxis], 3, axis=2)
    # img = np.where(cond, img, np.zeros_like(img))
    # print(np.unique(depth_array, return_counts=True))
    # cv2.imshow("mask", mask)
    mask = mask[:, :, 0] > 0
    cv2.waitKey(0)
    # print(mask.shape, np.unique(mask, return_counts=True))
    _ = gpe.estimate_from_depth_array(depth_array, mask, item=8)

    gpe._change_visualization(True)
    # Toothpaste
    depth_array = np.load(
        os.path.join("robot_cell", "packet", "data", "new_items", "4_depth_array.npy")
    )
    img = cv2.imread(
        os.path.join("robot_cell", "packet", "data", "new_items", "4_rgb.jpg")
    )

    print(img.shape, depth_array.shape)
    x_f = 790
    x_t = x_f + 424
    y_f = 510
    y_t = y_f + 210
    img = img[y_f:y_t, x_f:x_t, :]
    depth_array = depth_array[y_f:y_t, x_f:x_t]
    cv2.imshow("img", img)
    cv2.waitKey(0)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnt = tuple()
    mask = np.zeros_like(img)
    for i in range(3):
        imgray = img[:, :, i]
        ret, thresh = cv2.threshold(imgray, 35, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        p = img.copy()
        # cv2.drawContours(p, contours, -1, (0,255,0), 1)
        # cv2.imshow("img", p)
        # cv2.waitKey(0)
        mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    # cond = np.repeat((depth_array < 770)[:, :, np.newaxis], 3, axis=2)
    # img = np.where(cond, img, np.z eros_like(img))
    # print(np.unique(depth_array, return_counts=True))
    # cv2.imshow("mask", mask)
    mask = mask[:, :, 0] > 0
    # print(mask.shape, np.unique(mask, return_counts=True))
    _ = gpe.estimate_from_depth_array(depth_array, mask, item=8)

    # print("second")
    # gpe.estimate_from_images("rgb_image1.jpg", "depth_image1.png", r"robot_cell\packet\data\packet+hand")
    # mask = np.logical_not(mask)
    # point_relative, normal = gpe.estimate_from_depth_array(depth_array, mask)
    # # normal = np.array([0, 0, 1])
    # print(f"Estimated normal:\t {normal} in packet base")
    # R = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    # a, b, c = gpe._vectors2RPYrot(normal, R)
    # print(f"Angles between gripper base and normal: {a:.2f},  {b:.2f},  {c:.2f}")


if __name__ == "__main__":
    main()
