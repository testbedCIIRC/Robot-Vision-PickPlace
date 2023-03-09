import copy

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

from robot_cell.packet.packet_object import Packet


class PacketTracker:
    """
    Class for tracking packets between frames.
    """

    def __init__(
        self, maxDisappeared: int, guard: int = 250, maxCentroidDistance: int = 100
    ):
        """
        PacketTracker object constructor.

        Args:
            maxDisappeared (int): Maximum number of frames before deregister.
            guard (int): When packet depth is cropped, the resulting crop will have 'guard' extra pixels on each side.
            maxCentroidDistance (int): Maximal distance which packet can travel when it disappears in frame pixels.
        """

        self.nextObjectID = 0
        self.packets = OrderedDict()

        self.guard = guard
        self.maxCentroidDistance = maxCentroidDistance

        # Maximum consecutive frames a given object is allowed to be marked as "disappeared"
        self.maxDisappeared = maxDisappeared

    def register(self, packet: Packet, frame: np.ndarray):
        """
        Registers input item.

        Args:
            packet (Packet): Packet object to be registered.
            frame (np.ndarray): Image frame.
        """

        # When registering an object we use the next available object
        # ID to store the packet
        self.packets[self.nextObjectID] = packet
        self.packets[self.nextObjectID].disappeared = 0

        crop = self.get_crop_from_frame(self.packets[self.nextObjectID], frame)
        self.packets[self.nextObjectID].depth_maps = crop
        self.packets[self.nextObjectID].id = self.nextObjectID  # ! only for itemObject
        self.nextObjectID += 1

    def deregister(self, objectID: str):
        """
        Deregisters object based on id.

        Args:
            objectID (str): Key to deregister items in the objects dict.
        """

        # Save and return copy of deregistered packet data
        deregistered_packet = copy.deepcopy(self.packets[objectID])
        # To deregister an object ID we delete the object ID from our dictionary
        del self.packets[objectID]
        return deregistered_packet

    def get_crop_from_frame(self, packet: Packet, frame: np.ndarray):
        """
        Cuts packet out of the image frame.

        Args:
            packet (Packet): Packet object to be cut out.
            frame (np.ndarray): Image frame from which the packet should be cut out.

        Returns:
            np.ndarray: Packet cutout.
        """

        # Get packet specific crop from frame
        crop = frame[
            (packet.centroid_px.y - int(packet.height / 2) - self.guard) : (
                packet.centroid_px.y + int(packet.height / 2) + self.guard
            ),
            (packet.centroid_px.x - int(packet.width / 2) - self.guard) : (
                packet.centroid_px.x + int(packet.width / 2) + self.guard
            ),
        ]
        crop = np.expand_dims(crop, axis=2)
        return crop

    def update(
        self, detected_packets: list[Packet], frame: np.ndarray
    ) -> tuple[OrderedDict, list[Packet]]:
        """
        Updates the currently tracked detections.

        Args:
            detected_packets (list[Packet]): List containing detected packet objects to be tracked.
            frame (np.ndarray): Image frame.

        Returns:
            OrderedDict: Ordered dictionary with tracked detections.
            list[Packet]: List of packets that were deregistered.
        """

        deregistered_packets = []
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(detected_packets) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.packets.keys()):
                self.packets[objectID].disappeared += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.packets[objectID].disappeared > self.maxDisappeared:
                    deregistered_packet = self.deregister(objectID)
                    deregistered_packets.append(deregistered_packet)
            # return early as there are no centroids or tracking info
            # to update
            return self.packets, deregistered_packets

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.packets) == 0:
            for i in range(0, len(detected_packets)):
                self.register(detected_packets[i], frame)

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.packets.keys())
            objectCentroids = [
                self.packets[key].centroid_px for key in list(self.packets.keys())
            ]
            inputCentroids = [packet.centroid_px for packet in detected_packets]
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue
                # if D[row, col] > self.maxCentroidDistance:
                #     self.register(detected_packets[col], frame)
                #     continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]

                self.packets[objectID].centroid_px = detected_packets[col].centroid
                self.packets[objectID].angles.append(detected_packets[col].angles[0])
                self.packets[objectID].disappeared = 0

                crop = self.get_crop_from_frame(self.packets[objectID], frame)

                if crop.shape[0:2] == self.packets[objectID].depth_maps[:, :, 0].shape:
                    self.packets[objectID].depth_maps = np.concatenate(
                        (self.packets[objectID].depth_maps, crop), axis=2
                    )

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.packets[objectID].disappeared += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.packets[objectID].disappeared > self.maxDisappeared:
                        deregistered_packet = self.deregister(objectID)
                        deregistered_packets.append(deregistered_packet)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(detected_packets[col], frame)

        # return the set of trackable objects
        return self.packets, deregistered_packets
