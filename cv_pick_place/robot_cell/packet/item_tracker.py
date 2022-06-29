from math import sqrt
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class ItemTracker:
    def __init__(self, maxDisappeared, guard = 250, max_item_distance = 300):
        """
        ItemTracker object constructor.
    
        Parameters:
        maxDisappeared (int): Maximum number of frames before deregister.
        guard (int): When packet depth is cropped, the resulting crop will have 'guard' extra pixels on each side
        maxCentroidDistance (int): Maximal distance which packet can travel when it disappears in frame pixels
        """
        self.nextObjectID = 0
        self.packets = OrderedDict()

        self.guard = guard
        self.maxCentroidDistance = max_item_distance

        # Maximum consecutive frames a given object is allowed to be marked as "disappeared"
        self.maxDisappeared = maxDisappeared

        self.next_item_id = 0
        self.item_database = []
        self.max_item_distance = max_item_distance

    def register_item(self, item):
        item.id = self.next_item_id
        self.next_item_id += 1
        self.item_database.append(item)

    def deregister_item(self, id):
        for tracked_item_index, tracked_item in enumerate(self.item_database):
            if tracked_item.id == id:
                del self.item_database[tracked_item_index]
                break

    def update_item(self, new_item, tracked_item):
        if tracked_item.id != new_item.id:
            print("[WARNING] Tried to update two items with different IDs together")
            return
        
        # Update parameters
        tracked_item.centroid = new_item.centroid
        tracked_item.disappeared = 0

        return tracked_item

    def update_item_database(self, labeled_item_list):
        # Increment disappeared frame on all items
        for tracked_item in self.item_database:
            tracked_item.disappeared += 1

        for labeled_item_index, labeled_item in enumerate(labeled_item_list):
            # Register new item
            if labeled_item.id == None:
                self.register_item(labeled_item)
                continue

            # Update exitsing item data
            for tracked_item_index, tracked_item in enumerate(self.item_database):
                if labeled_item.id == tracked_item.id:
                    self.item_database[tracked_item_index] = self.update_item(labeled_item, tracked_item)
                    break
        
        for tracked_item in self.item_database:
            if tracked_item.disappeared > self.maxDisappeared:
                self.deregister_item(tracked_item.id)

    def track_items(self, detected_item_list):
        """
        Labels input items with IDs from tracked item database,
        depending on distance.
        """
        labeled_item_list = detected_item_list

        for detected_item_index, detected_item in enumerate(detected_item_list):
            # For every newly detected item, store distance to all tracked items
            minimal_distance = None
            minimal_dist_index = None

            for tracked_item_index, tracked_item in enumerate(self.item_database):
                # Compute distance
                dist = sqrt((detected_item.centroid[0] - tracked_item.centroid[0]) ** 2 + (detected_item.centroid[1] - tracked_item.centroid[1]) ** 2)

                if minimal_distance is None or dist < minimal_distance:
                    minimal_distance = dist
                    minimal_dist_index = tracked_item_index
            
            if minimal_dist_index is not None and minimal_distance < self.max_item_distance:
                labeled_item_list[detected_item_index].id = self.item_database[tracked_item_index].id


        return labeled_item_list

    def register(self, packet, frame):
        """
        CRegisters input item.
    
        Parameters:
        packet (object): packet object to be registered.
        frame (numpy.ndarray): image frame.

        """
        # When registering an object we use the next available object
        # ID to store the packet
        self.packets[self.nextObjectID] = packet
        self.packets[self.nextObjectID].disappeared = 0

        crop = self.get_crop_from_frame(self.packets[self.nextObjectID], frame)
        self.packets[self.nextObjectID].depth_maps = crop

        self.nextObjectID += 1

    def deregister(self, objectID):
        """
        Deregisters object based on id.
    
        Parameters:
        objectID (str): Key to deregister items in the objects dict.

        """
        # Save and return copy of deregistered packet data
        deregistered_packet = self.packets[objectID]
        # To deregister an object ID we delete the object ID from our dictionary
        del self.packets[objectID]
        return deregistered_packet

    def get_crop_from_frame(self, packet, frame):
        # Get packet specific crop from frame
        crop = frame[(packet.centroid[1] - int(packet.height / 2) - self.guard):(packet.centroid[1] + int(packet.height / 2) + self.guard),
               (packet.centroid[0] - int(packet.width / 2) - self.guard):(packet.centroid[0] + int(packet.width / 2) + self.guard)]
        crop = np.expand_dims(crop, axis=2)
        return crop

    def update(self, detected_packets, frame):
        """
        Updates the currently tracked detections.
    
        Parameters:
        detected_packets (list): List containing detected packet objects to be tracked.
    
        Returns:
        OrderedDict: Ordered dictionary with tracked detections.
        list: packets that were deregistered.

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
                # if D[row, col] > self.maxCentroidDistance:
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
