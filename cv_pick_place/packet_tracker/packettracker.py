from scipy.spatial import distance as dist
from collections import OrderedDict
import copy
import numpy as np

# Crop guard region
# When packet depth is cropped, the resulting crop will have 'guard' extra pixels on each side
guard = 250

# Maximal distance which packet can travel when it disappears in frame pixels
# When a distance is greater than this, new packet is created instead
maxCentroidDistance = 100


class PacketTracker:
    def __init__(self, maxDisappeared):
        self.nextObjectID = 0
        self.packets = OrderedDict()

        # Maximum consecutive frames a given object is allowed to be marked as "disappeared"
        self.maxDisappeared = maxDisappeared

    def register(self, packet, frame):
        # When registering an object we use the next available object
        # ID to store the packet
        self.packets[self.nextObjectID] = packet
        self.packets[self.nextObjectID].disappeared = 0

        crop = self.get_crop_from_frame(self.packets[self.nextObjectID], frame)
        self.packets[self.nextObjectID].depth_maps = crop

        self.nextObjectID += 1

    def deregister(self, objectID):
        # Save and return copy of deregistered packet data
        deregistered_packet = copy.deepcopy(self.packets[objectID])
        # To deregister an object ID we delete the object ID from our dictionary
        del self.packets[objectID]
        return deregistered_packet

    def get_crop_from_frame(self, packet, frame):
        # Get packet specific crop from frame
        crop = frame[(packet.centroid[1] - int(packet.height / 2) - guard):(packet.centroid[1] + int(packet.height / 2) + guard),
               (packet.centroid[0] - int(packet.width / 2) - guard):(packet.centroid[0] + int(packet.width / 2) + guard)]
        crop = np.expand_dims(crop, axis=2)
        return crop

    def update(self, detected_packets, frame):
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
            objectCentroids = [self.packets[key].centroid for key in list(self.packets.keys())]
            inputCentroids = [packet.centroid for packet in detected_packets]
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
                # if D[row, col] > maxCentroidDistance:
                #     self.register(detected_packets[col], frame)
                #     continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]

                self.packets[objectID].centroid = detected_packets[col].centroid
                self.packets[objectID].angles.append(detected_packets[col].angles[0])
                self.packets[objectID].disappeared = 0

                crop = self.get_crop_from_frame(self.packets[objectID], frame)

                if crop.shape[0:2] == self.packets[objectID].depth_maps[:, :, 0].shape:
                    self.packets[objectID].depth_maps = np.concatenate((self.packets[objectID].depth_maps, crop), axis=2)

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