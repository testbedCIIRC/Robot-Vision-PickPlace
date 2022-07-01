import numpy as np

# Class containing relevant packet data for convenience
class Packet:
    def __init__(self, box = np.empty(()), pack_type = None, centroid = (0, 0), 
                    centroid_depth = 0, angle = 0, width = 0, height = 0,
                     ymin= 0, ymax= 0, xmin= 0, xmax= 0, encoder_position = 0):
        """
        Constructs packet objects.
    
        Parameters:
        box (numpy.ndarray): packet bounding box.
        pack_type (bool): class of detected packet.
        centroid (tuple): centroid of packet.
        centroid_depth (int): depth of centroid from depth frame.
        angle (float): angle of bounding box.
        width (int): width of bounding box.
        height (int): height of bounding box.
        encoder_position (float): position of encoder.
    
        """
        self.id = None

        # Tuple of 2 numbers describing center of the packet
        self.centroid = centroid

        # List of angles for averaging
        self.angles = [angle]
        self.angle = angle

        # Width and height of packet bounding box
        self.width = width
        self.height = height

        self.yminbbx = ymin
        self.ymaxbbx = ymax
        self.xminbbx = xmin
        self.xmaxbbx = xmax

        # Numpy array of cropped depth maps
        self.depth_maps = np.empty(())
        self.color_frames = np.empty(())

        # Number of frames the packet has disappeared for
        self.disappeared = 0

        self.time_of_disappearance = None

        self.box = box

        self.pack_type = pack_type

        # Encoder data
        self.starting_encoder_position = encoder_position

        # Used for offsetting centroid position calculated using encoder
        self.first_centroid_position = centroid


        self.marked_for_picking = False
    def getCentroidFromEncoder(self, encoder_position):
        """
        Computes actual centroid of packet from the encoder data.
    
        Parameters:
       
        encoder_pos (float): current encoder position.
        

        Returns:
        float: Updated x, y packet centroid.
    
        """
        # k = 0.8299  # 640 x 480
        # k = 1.2365  # 1280 x 720
        k = 1.8672  # 1440 x 1080
        # k = 1.2365  # 1080 x 720
        return (int(k * (encoder_position - self.starting_encoder_position) + self.first_centroid_position[0]), self.centroid[1])

    # Get centroid in world coordinates
    def getCentroidInWorldFrame(self, homography):
        """
        Converts centroid from image coordinates to real world coordinates.
    
        Parameters:
       
        homography (numpy.ndarray): homography matrix.
        

        Returns:
        float: Updated x, y packet centroid in world coordinates.
    
        """
        centroid_robot_frame = np.matmul(homography, np.array([self.centroid[0], self.centroid[1], 1]))

        packet_x = centroid_robot_frame[0] * 10
        packet_y = centroid_robot_frame[1] * 10
        return packet_x, packet_y