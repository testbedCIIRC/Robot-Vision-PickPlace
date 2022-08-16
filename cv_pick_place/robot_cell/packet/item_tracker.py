from math import sqrt
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class ItemTracker:
    def __init__(self, max_disappeared_frames = 10, guard = 250, max_item_distance = 200):
        """
        ItemTracker object constructor.
    
        Parameters:
        max_disappeared_frames (int): Maximum number of frames before deregister.
        guard (int): When packet depth is cropped, the resulting crop will have 'guard' extra pixels on each side
        max_item_distance (int): Maximal distance which packet can travel when it disappears in frame pixels

        """
        self.item_database = []
        self.next_item_id = 0

        self.max_item_distance = max_item_distance
        self.max_disappeared_frames = max_disappeared_frames
        self.guard = guard

    def register_item(self, item):
        """
        Adds new item into database

        Parameters:
        item (Item): New item object which should be added to the database

        """
        item.id = self.next_item_id
        item.disappeared = 0
        self.next_item_id += 1
        self.item_database.append(item)

    def deregister_item(self, id):
        """
        Removes item with matching id from tracking database

        Parameters:
        id (int): New item object whose parameters are transferred

        """
        for tracked_item_index, tracked_item in enumerate(self.item_database):
            if tracked_item.id == id:
                del self.item_database[tracked_item_index]
                break

    def update_item(self, new_item, tracked_item):
        """
        Updates parameters of single tracked item with those of new item

        Parameters:
        new_item (Item): New item object whose parameters are transferred
        tracked_item (Item): Item object whose parameters are updated
        
        Output:
        tracked_item (Item): Updated tracked item object

        """
        if tracked_item.id != new_item.id:
            print("[WARNING] Tried to update two items with different IDs together")
            return
        
        # NEW parameters
        tracked_item.centroid_px = new_item.centroid_px
        tracked_item.centroid_mm = new_item.centroid_mm

        tracked_item.width_bnd_px = new_item.width_bnd_px
        tracked_item.width_bnd_mm = new_item.width_bnd_mm
        tracked_item.height_bnd_px = new_item.height_bnd_px
        tracked_item.height_bnd_mm = new_item.height_bnd_mm

        tracked_item.add_angle_to_average(new_item.avg_angle_deg)

        # OLD parameters
        tracked_item.width = new_item.width
        tracked_item.height = new_item.height
        tracked_item.centroid = new_item.centroid
        tracked_item.disappeared = 0
        tracked_item.box = new_item.box

        return tracked_item

    def update_item_database(self, labeled_item_list):
        """
        Update tracked item database using labeled detected items.
        When item has id of None, it is registered as new item.
        When item has same id already tracked item, the tracked item is updated with detected parameters.

        Parameters:
        labeled_item_list (List[Item]): List of detected Item objects with id == id of nearest tracked item
        
        """

        # Increment disappeared frame on all items
        for tracked_item in self.item_database:
            tracked_item.disappeared += 1

        for labeled_item in labeled_item_list:
            # Register new item
            if labeled_item.id == None:
                self.register_item(labeled_item)
                continue

            # Update exitsing item data
            for tracked_item_index, tracked_item in enumerate(self.item_database):
                if labeled_item.id == tracked_item.id:
                    self.item_database[tracked_item_index] = self.update_item(labeled_item, tracked_item)
                    break
        
        # Check for items ready to be deregistered
        for tracked_item in self.item_database:
            if tracked_item.disappeared > self.max_disappeared_frames:
                self.deregister_item(tracked_item.id)

    def track_items(self, detected_item_list):
        """
        Labels input items with IDs from tracked item database, depending on distance.

        Parameters:
        detected_item_list (List[Item]): List of detected Item objects with id == None

        Output:
        labeled_item_list (List[Item]): List of detected Item objects with id == id of nearest tracked item

        """
        labeled_item_list = detected_item_list
        # if no packets are being detected or tracked
        if len(detected_item_list) == 0 or len(self.item_database) == 0:
            return labeled_item_list
        else:
            # create a list of tracked and detected centroids
            trackedCentroids = [item.centroid for item in self.item_database]
            detectCentroids = [item.centroid for item in detected_item_list]
            # compute the distance between each pair of items
            distances = dist.cdist(np.array(trackedCentroids), np.array(detectCentroids))
            # sort tracked items (rows) by minimal distance
            tracked = distances.min(axis=1).argsort()
            # sort detected items (columns) by minimal distance
            detected = distances.argmin(axis=1)[tracked]

            usedTracked = set()
            usedDetected = set()
            # loop over the combination of the (row, column) index, starting from minimal distances
            for (trac, det) in zip(tracked, detected):
                # ignore already used items
                if trac in usedTracked or det in usedDetected:
                    continue
                # if assigned distance is too far, ignore it
                if distances[trac, det] > self.max_item_distance:
                    continue
                # assign id to detected item
                labeled_item_list[det].id = self.item_database[trac].id

                # indicate which items were used
                usedTracked.add(trac)
                usedDetected.add(det)

        return labeled_item_list
