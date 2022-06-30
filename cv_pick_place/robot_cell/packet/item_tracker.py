from math import sqrt
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class ItemTracker:
    def __init__(self, max_disappeared_frames = 10, guard = 250, max_item_distance = 300):
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
        self.maxDisappeared = max_disappeared_frames

        self.next_item_id = 0
        self.item_database = []
        self.max_item_distance = max_item_distance
        self.max_disappeared_frames = max_disappeared_frames

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
        
        # Update parameters
        tracked_item.centroid = new_item.centroid
        tracked_item.disappeared = 0

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
