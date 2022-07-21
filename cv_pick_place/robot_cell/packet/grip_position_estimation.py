from tkinter.messagebox import NO
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

from robot_cell.packet.packet_object import Packet


class GripPositionEstimation():
    def __init__(self, visualize: bool = False, verbose: bool = False,
                 center_switch: str = "mass", gripper_radius: float = 0.05,
                 gripper_ration: float = 0.8, runs_max_number: int = 100,
                  height_th: float = -0.76, num_bins: int = 20):
        """
        Initializes class for predicting optimal position for the gripper
        
        Parameters:
        visualize (bool): if True will visualize the results
        verbose (bool): if True  will write some explanetions of whats happening
        center_switch (str): "mass" or "height" - defines the center of the gripper
        gripper_radius (float): radius of the gripper
        gripper_ration (float): ratio of gripper radius for dettecting the gripper annulus
        runs_max_number (int): maximal number of tries to estimate the optimal position
        height_th (float): distance between camera and belt
        num_bins (int): number of bins for height thresholding (20 is good enough, 10 works as well)
        """
        self.visualization = visualize
        self.verbose = verbose
        # assert center_switch in ["mass", "height"], "center_switch must be 'mass' or 'height'"
        self.center_switch = center_switch
        # assert gripper_radius > 0, "gripper_radius must be positive"
        self.gripper_radius = gripper_radius
        # assert gripper_ration > 0, "gripper_ration must be positive"
        self.gripper_ratio = gripper_ration # Ration for computing gripper annulus
        
        # Pointcloud
        self.pcd = None 
        self.points = None
        self.normals = None

        # Histograms
        self.th_idx = 0
        self.hist = None
        self.th_val = height_th # Height threshold - defautly measured physically
        self.num_bins = num_bins

        
        self.max_runs = runs_max_number # Max number for tries for the estimation of best pose
        self.run_number = 0 # current run
        self.spike_threshold = 0.04 # threshold for spike in the circlic neighborhood


    def _create_pcd_from_rgb_depth_frames(self, rgb_name: str, depth_name: str,
                                      path: str = "") -> o3d.geometry.PointCloud:
        """
        Loads open3d PointCloud from rgb image and depth image stored in path into self.pcd
        
        Parameters:
        rgb_name (str): name of the rgb image(.jpg)
        depth_name (str): name of the depth image(.png)
        path (str): path to the images

        Returns: 
        o3d.geometry.PointCloud: pointcloud
        """

        rgb = o3d.io.read_image(os.path.join(path, rgb_name))
        depth = o3d.io.read_image(os.path.join(path, depth_name))
        rgbd= o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd])

        # FIXME: Remove  later Was here for visualization
        # d = self._load_numpy_depth_array_from_png(os.path.join(path, depth_name)) 
        # p = self._create_pcd_from_depth_array(d)
        return pcd


    def _load_numpy_depth_array_from_png(self, depth_name:str) -> np.ndarray:
        """
        Load png image and converts it to the numpy ndarray
        # NOTE: wont be propably used in the end just for testing
        Parameters:
        depth_name (str): Path to the png. file with depth

        Returns:
        np.ndarray: Depth stored in 2D numpy ndarray
        """
        image = Image.open(depth_name)
        return np.array(image)


    def _create_pcd_from_depth_array(self, depth_frame:np.ndarray
                                   ) -> o3d.geometry.PointCloud:
        """
        Loads open3d PointCloud from depthframe

        Parameters:
        depth_frame (np.ndarray): Depth values

        Returns:
        o3d.geometry.PointCloud: Pointcloud
        """
        depth = o3d.geometry.Image(np.ascontiguousarray(depth_frame).astype(np.uint16))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd])
        return pcd


    def _pcd_down_sample(self, voxel_size: float = 0.01):
        """
        Downsample PointCloud with given voxel size
        
        Parameters:
        voxel_size (float): voxel size for downsampling
        """
        pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        self.pcd = pcd


    def _get_points_and_estimete_normals(self) -> tuple[np.ndarray]:
        """
        Estimates normals from PointCloud and saves both points coordinates and
        their respective normals

        Returns:
        tuple[np.ndarray, np.ndarray] : vetex points and their normals
        """
        self.pcd.estimate_normals()
        self.pcd_center = self.pcd.get_center()
        self.max_bound = self.pcd.get_max_bound()
        return np.asarray(self.pcd.points), np.asarray(self.pcd.normals)


    def _visualize_frame(self,points_dict:dict
                         ) -> None:
        """
        Visualize points in 2D, firstly all points secondly all sets points from
        points_dict, Uses key as legend and plots the numpy array values,

        Parameters:
        points_dict (dict): Dictionary with keys used as legend and values numpy:ndarrays
        """
        plt.figure(2)
        plt.scatter(self.points[:, 0], self.points[:, 1])
        legend = ('All points',)
        for key, vals in points_dict.items():
            if len(vals.shape) == 2:
                plt.scatter(vals[:, 0], vals[:, 1])
            else:
                plt.scatter(vals[0], vals[1])
            legend += (key,)

        plt.legend(legend)
        plt.title("Points 2D")
        plt.show()


    def _visualize_histogram(self, n_bins:int) -> None:
        """
        Visualizes histogram with selected threshold
        """
        
        fig, ax = plt.subplots()
        N, bins, patches = ax.hist(self.points[:, 2], bins=n_bins)
        for i in range(0, self.th_idx):
            patches[i].set_facecolor('#1f77b4')
        for i in range(self.th_idx, len(patches)):
            patches[i].set_facecolor('#ff7f0e')
        plt.title('Depth histogram')
        plt.xlabel('z [m]')
        plt.ylabel('count [-]')
        plt.show()


    def _hist_first_peak(self, th_count: int = 50) -> float:
        """
        Finds first peak in histogram. NOT SMART
        
        Parameters:
        th_count (int):threshold value for finding the first peak with 
        atleast 50 count

        Returns:
        float: Height threshold between belt and object
        """
        # assert len(self.hist.shape) == 1, "only 1D histogram"
        old = -np.inf
        s = 0
        for e, h in enumerate(self.hist):
            s += h
            if h < old and s > th_count:
                return e
            old = h


    def _compute_histogram_threshold(self, values: np.ndarray,
                                     number_of_bins: int = 20) -> None:
        """
        Finds height threshold via histogram stores it in itself

        Parameters:
        values (np.ndarray): values to be binned
        number_of_bins (int): number of bins to be used in histogram

        """
        # assert len(values.shape) == 1, "Values should be 1D"

        h = np.histogram(values, bins=number_of_bins)
        self.hist = h[0]
        i = self._hist_first_peak()
        self.th_idx = i + number_of_bins//10
        # self.th_val = -0.76 Some constant used for the 
        self.th_val = h[1][self.th_idx]


    def _get_center_of_mass(self, points: np.ndarray) -> np.ndarray:
        """
        Get center of mass from points via approximation of mass with height (z axis)

        Parameters:
        points (np.ndarray): 3D points for computation

        Returns:
        tuple[np.ndarray, np.ndarray]: geometric center and center of mass
        """
        c_mass = np.average(points[:, :2], axis=0, weights=points[:, 2]-np.min(points[:, 2]))
        dists = np.linalg.norm(points[:, :2] - c_mass, axis=1)
        idx = np.argmin(dists)
        c_mass = np.hstack((c_mass, self.points[idx, 2]))

        return c_mass


    def _anuluss_mask(self, center: np.ndarray) -> np.ndarray:
        """
        Returns mask of pointes which are in annulus

        Parameters:
        center (np.ndarray):center point of the annulus, 
        
        Returns:
        np.ndarray: binary mask
        """
        l2 = np.linalg.norm(self.points-center, axis=1)
        s = l2 >= self.gripper_radius*self.gripper_ratio
        b = l2 <= self.gripper_radius
        return b*s


    def _circle_mask(self, center: np.ndarray, radius:float = 0.05
                     ) -> np.ndarray:
        """
        Returns binary mask of points which are are closer to center than radius

        Parameters:
        center (np.ndarray): shape (2,)
        radius (float): radius of the circle

        Returns:
        np.ndarray: binary mask
        """
        l2 = np.linalg.norm(self.points-center, axis=1)
        return l2 <= radius


    def _fit_plane(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits plane to given points
        
        Parameters:
        points (np.ndarray): points to be fitted to plane

        Returns:
        tuple[np.ndarray, np.ndarray]: plane center and plane normal
        """
        # assert points.shape[1] == 3, "Points should have size 3(x,y,z)"
        # assert len(points.shape) == 2, "Points should be 2D"
        c = points.mean(axis=0)
        A = points - c
        M = np.dot(A.T, A)
        return c, np.linalg.svd(M)[0][:, -1]


    def _get_candidate_point(self, center: np.ndarray, filter_mask: np.ndarray,
                             blacklist:np.ndarray) -> np.ndarray:
        """
        Selects point closest to center, which is not blacklisted
        
        Parameters:
        center (np.ndarray): center of mass
        filter_mask (np.ndarray): binary mask of fitered points (that makes the object)
        blacklist (np.ndarray): binary mask of points that are not yet blacklisted
        """
        candidate_mask = np.logical_and(filter_mask, blacklist)
        candidate_points = self.points[candidate_mask,:]
        l2 = np.linalg.norm(self.points[candidate_mask,:2] - center[:2], axis=1)
        idx = np.argmin(l2)
        return candidate_points[idx, :]


    def _check_point_validity(self, point: np.ndarray) -> bool:
        """
        Checks if point is valid. No height spike in circlic neighborhood,
        Check if part of the outer circle is not on the conveyer belt
        
        Parameters:
        point (np.ndarray): point to be checked

        Returns:
        bool: True if point is valid, False otherwise
        """
        # l2 (Euclidian) distance
        l2 = np.linalg.norm(self.points[:,:]-point[:], axis=1)
        mask_inner = l2 < self.gripper_radius * self.gripper_ratio

        # If no points are in given l2 distatnce
        if not np.any(mask_inner):
            return False

        max_point = np.argmax(self.points[mask_inner, 2])
        min_point = np.argmin(self.points[mask_inner, 2])
        valid_max = np.abs(self.points[max_point, 2] - point[2]) < self.spike_threshold
        valid_min = np.abs(self.points[min_point, 2] - point[2]) < self.spike_threshold

        ann_mask = self._anuluss_mask(point)
        ann_points = self.points[ann_mask, :]
        lowest_point = np.argmin(ann_points[:,2])
        # print(ann_points[lowest_point, 2])

        valid_conv = ann_points[lowest_point, 2] > self.th_val   

        valid = valid_max and valid_min and valid_conv
        #BUG: Somwhere always tries the same Point with dept_image2.png Something wrong with blacklisting
        if self.verbose:
            print(f"[INFO]: Point {point} is valid: {valid}")
            if not valid_max:
                print(f"\tReason - Spike:\tPoint {self.points[max_point, :]} difference in height is: {np.abs(self.points[max_point,2] - point[2])} spike threshold: { self.spike_threshold}")
            if not valid_min:
                print(f"\tReason - Spike:\tPoint {self.points[min_point, :]} difference in height is: {np.abs(self.points[min_point,2] - point[2])} spike threshold: {self.spike_threshold}")            
            if not valid_conv:
                print(f"\tReason - Belt:\t Part of circle with point as center is on conveyer belt, i.e. point :{ann_points[lowest_point, :]} Height threshold: {self.th_val}")

        return valid


    def _expand_blacklist(self, point: np.ndarray, blacklist: np.ndarray,
                          radius: float=0.01) -> np.ndarray:
        """
        Expands blacklist by points which are closer than radius to the given
        point

        Parameters:
        point (np.ndarray): center point
        blacklist (np.ndarray): binary mask, True if it can be used, False if it is blacklisted
        radius (float): threshold

        Returns:
        np.ndarray: updated binary mask for blacklist
        """
        bl = np.logical_not(self._circle_mask(point, radius))
        return np.logical_and(blacklist, bl)   


    def _detect_point_from_pcd(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Detects points from point cloud. Firstly calculates center of mass,
        check its validity, if its good returns it, if not blacklist
        neighborhood, and selects clossest point to center of mass.
        Checks again

        Returns:
        tuple[np.ndarray, np.ndarray]: plane center and plane normal (x,y,z)
        """
        self._pcd_down_sample()
        self.points, self.normals = self._get_points_and_estimete_normals()

        # o3d.visualization.draw_geometries([self.pcd])

        self._compute_histogram_threshold(self.points[:, 2], self.num_bins)

        if self.visualization:
            self._visualize_histogram(self.num_bins)

        packet_mask = self.points[:, 2] >= self.th_val
        filtered_points = self.points[packet_mask, :]
        blacklist = np.full((self.points.shape[0],), True)   

        if self.center_switch == 'mass':
            center = self._get_center_of_mass(filtered_points)
        elif self.center_switch == 'height':
            center = self.points[np.argmax(filtered_points[:,2]),:]            
        
        n_mask = self._anuluss_mask(center)
        plane_c, plane_n = self._fit_plane(self.points[n_mask, :])
        valid = self._check_point_validity(center)

        # Returns original point as optimal if it is valid
        if valid:
            if self.verbose:
                print(f"[INFO]: Picked original point as it is valid")
            if self.visualization:
                viz_dict = {
                    "Height filtered points": filtered_points,
                    "Neighborhood": self.points[n_mask, :],
                    "center point " + self.center_switch: center,
                    "center plane": plane_c,
                }
                self._visualize_frame(viz_dict)
            return center, plane_n

        blacklist = self._expand_blacklist(center, blacklist)
        while self.run_number < self.max_runs:
            # t1 = datetime.now()
            c_point = self._get_candidate_point(center, packet_mask, blacklist)
            valid = self._check_point_validity(c_point)
            if not valid:
                blacklist = self._expand_blacklist(center, blacklist)
                # t2 = datetime.now()
                # print(str(t2-t1))
                self.run_number += 1
                continue

            n_mask = self._anuluss_mask(c_point)
            neighbourhood_points = self.points[n_mask, :]
            plane_c, plane_n = self._fit_plane(neighbourhood_points)

            if self.visualization:
                viz_dict = {
                    "Height filtered points": filtered_points,
                    "Neighborhood": self.points[n_mask, :],
                    "center point " + self.center_switch: center,
                    "center plane": plane_c,
                }
                self._visualize_frame(viz_dict)

            return center, plane_n

        # Maybe replace with some warning
        if self.verbose:
            print(f"[WARN]: Could not find the valid point in given number of tries: {self.max_runs}. Returning None")
        return None, None


    def _get_relative_coordinates(self, point: np.ndarray, anchor: np.ndarray
                                  ) -> np.ndarray:
        """
        Returns relative coordinates of point from center of gripper

        Parameters:
        point (np.ndarray): point to be transformed
        anchor (np.ndarray): relative coordinates of anchor point(center(px,py))

        Returns: 
        np.ndarray: relative coordinates of point to anchor and height(px,py,z)
        """
        max_bound = self.pcd.get_max_bound()
        min_bound = self.pcd.get_min_bound()
        db = max_bound - min_bound
        point_rel = (point[:2] - min_bound[:2]) / db[:2]
        point_anchor_rel = point_rel - anchor

        return np.hstack((point_anchor_rel, point[2]))


    def _vector2angles(self, vector:np.ndarray,
                       rotation_matrix: np.ndarray=np.array([[-1.0, 0.0, 0.0],
                                                             [0.0, -1.0, 0.0],
                                                             [0.0, 0.0, 1.0]])
                        ) -> np.ndarray[float]:
        """
        Changes vector for approach by the gripper and converts it into angles
        for the gripper

        Parameters:
        vector (np.ndarray): vector to be converted, expects unit vector
        rotation_matrix (np.ndarrray): Rotation matrix between bases

        Returns
        tuple[float]: Angles ax, ay, az in degrees
        """
        # Opposite direction for gripper aproach
        normal_o = -1.0 * vector
        # Transform between bases of depth image and gripper
        g_normal = rotation_matrix @ normal_o

        # Converts vector to base angles
        ax = np.arctan2(np.linalg.norm(g_normal[1:]), g_normal[0]) # a = ...
        ay = np.arctan2(np.linalg.norm(g_normal[0::2]), g_normal[1]) # b = ...
        az = np.arctan2(np.linalg.norm(g_normal[:2]), g_normal[2]) # c = ...
        angles = np.array([ax, ay, az])
        return np.rad2deg(angles)

    def estimate_from_images(self, rgb_image_name: str, depth_image_name: str,
                             path: str="", anchor: np.array=np.array([.5, .5])
                             ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates point and normal from given images
        (The idea is to call this function and it returns everything  needed)

        Parameters:
        rgb_image_name (str): name of rgb image
        depth_image_name (str): name of depth image
        path (str): path to images
        anchor (np.ndarray): relative coordinates of anchor point(center(px,py))

        Returns:
        tuple[np.ndarray, np.ndarray]: point coordinates and its normal
        """
        # FIXME: Remove just testing wheter both methods are the same
        # print(type(os.path.join(path, depth_image_name)))
        # d_frame = self._load_numpy_depth_array_from_png(os.path.join(path, depth_image_name))
        # pcd_f = self._create_pcd_from_depth_array(d_frame)
        # pcd_f = pcd_f.voxel_down_sample(voxel_size=0.01)
        # pcd_f.estimate_normals()
        # p_f, n_f = np.asarray(pcd_f.points), np.asarray(pcd_f.normals)
        
        self.pcd =  self._create_pcd_from_rgb_depth_frames(rgb_image_name, depth_image_name, path)
        # self.pcd = self.pcd.voxel_down_sample(voxel_size=0.01)
        # self.pcd.estimate_normals()
        # p_i, n_i = np.asarray(self.pcd.points), np.asarray(self.pcd.normals)
        # print(p_f.shape, n_f.shape)
        # print(p_i.shape, n_i.shape)
        # print((p_f-p_i).sum()) # Yeah its 0.0- meaning it should be same
        # return None, None
        center, normal = self._detect_point_from_pcd()
        if center is None:
            print(f"[INFO]: Did not found optimal point")
            return center, normal
        relative = self._get_relative_coordinates(center, anchor)

        return relative, normal


    def estimate_from_depth_array(self, depth_array: np.ndarray,
                                  anchor: np.ndarray = np.array([.5, .5])
                                  ) -> tuple[np.ndarray]:
        """
        Estimates optimal point from depth array

        Parameters:
        depth_array (np.ndarray): Depth values
        anchor (np.ndarray): Relative coordinates of anchor point(center(px,py))
        
        Returns:
        tuple[np.ndarray]: Point and normal for picking
        """
        self.pcd = self._create_pcd_from_depth_array(depth_array)
        center, normal = self._detect_point_from_pcd()
        if center is None:
            return None, None
        
        relative = self._get_relative_coordinates(center, anchor)
        return relative, normal

    def estimate_from_packet(self, packet: Packet, z_lim:float,
                             conv2cam_dist: float = 777.0, 
                             coord_limit:tuple[float]= (600, 1800, 75, 470)):        
        """
        Estimates optimal point of the packet

        Parameters:
        packet (Packet): Packet for the estimator
        z_lim (float): Limit for the robot gripper for not coliding
        conv2cam_dist (float): Distance from conveyer to camera in mm
        coord_limit (tupple[float]): 

        Returns:
        TBD, Maybe dict as it is in main
        trajectory_dict = {
                    'x': pick_pos_x,
                    'y': pick_pos_y,
                    'rot': gripper_rot,
                    'packet_type': packet_type,
                    'x_offset': 0,
                    'pack_z': pick_pos_z
                    }
        Maybe named tupple 
        robot_coords = namedtuple('Coords', 'x y z a b c')
        """
        depth_frame = packet.avg_depth_crop
        pack_x, pack_y = packet.centroid_mm
        pack_z = z_lim
        ax, ay, az = None, None, None
        if depth_frame is not None:
            point_relative, normal = self.estimate_from_depth_array(depth_frame)
            # TODO: Recompute into coordinates normal given by packet
            dx, dy, z = point_relative

            # p_px, p_py = packet.centroid_px
            p_mx, p_my = packet.centroid_mm

            pack_x, pack_y = p_mx + dx*packet.width_bnd_mm, p_my + dy*packet.height_bnd_mm
            # Converts m to mm and shifts by the conv2cam_dist
            pack_z = abs(z - conv2cam_dist) * 1000.0 + conv2cam_dist
            
            # Rotation between the bases of the pcd and the gripper
            R = np.array([[-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0]])
            
            ax, ay, az = self._vector2angles(normal, R)
            
        else:
            print(f"[WARN]: Average depth frame is None. Returning centroid and limit depth")
            


        # NOTE: Sanizizes coordinates into permateble ones
        # Dont know wheater it should be here
        x_min, x_max, y_min, y_max = coord_limit
        pack_x = np.clip(pack_x, x_min, x_max)
        pack_y = np.clip(pack_y, y_min, y_max)
        
        # Create output
        trajectory_dict = {
                    'x': pack_x,
                    'y': pack_y,
                    'rot': 0, # TODO Might be all repleced by angles for the robot
                    'packet_type': packet.pack_type,
                    'x_offset': 0,
                    'pack_z': pack_z
                    }
        return trajectory_dict


def main():
    # Demo how to work with class

    # Size of the triangle edge and radius of the circle
    triangle_edge = 0.085  # in meters
    gripper_radius = triangle_edge/np.sqrt(3)

    # Creating new class with given params
    gpe = GripPositionEstimation(visualize=True, verbose =True, center_switch="mass", gripper_radius=gripper_radius, gripper_ration=0.8)
    depth_array = gpe._load_numpy_depth_array_from_png(os.path.join("cv_pick_place","robot_cell","packet", "data", "depth_image_crop.png"))
    gpe.estimate_from_depth_array(depth_array)
    # # Estimating point and normal from color and depth images
    # point, normal = gpe.estimate_from_images("color_image_crop.jpg", "depth_image_crop.png",
    #                                                                     path=os.path.join("cv_pick_place","robot_cell","packet", "data"), anchor=np.array([.5, .5]))
    # # point, normal = gpe.estimate_from_images("rgb_image2.jpg", "depth_image2.png",
    # #                                                                     path="data", anchor=np.array([.5, .5]))
    # if point is not None:
    #     print(f"[INFO]: Picked_point\t{point}\t picked_normal\t {normal} norm = {np.linalg.norm(normal)}")

    # Estimate point and normal from anouther image pairs
    # gpe.estimate_optimal_point_and_normal_from_images("rgb_image1.jpg", "depth_image1.png", path="data")
    # print("picked_point", point)
    # print("picked_normal", normal, np.linalg.norm(normal))


if __name__ == "__main__":
    main()
