import numpy as np

# Class containing relevant packet data for convenience
class Packet:
    def __init__(self, box = np.empty(()), pack_type = None,centroid=(0, 0), angle=0, width=0, height=0):
        # Tuple of 2 numbers describing center of the packet
        self.centroid = centroid

        # List of angles for averaging
        self.angles = [angle]
        self.angle = angle

        # Width and height of packet bounding box
        self.width = width
        self.height = height

        # Numpy array of cropped depth maps
        self.depth_maps = np.empty(())

        # Number of frames the packet has disappeared for
        self.disappeared = 0

        self.box = box

        self.pack_type = pack_type