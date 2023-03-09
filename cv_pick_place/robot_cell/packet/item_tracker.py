import numpy as np
from scipy.spatial import distance as dist

from robot_cell.packet.packet_object import Packet


class ItemTracker:
    """
    Class for tracking packets between frames.
    """

    def __init__(
        self,
        max_disappeared_frames: int = 10,
        guard: int = 250,
        max_item_distance: int = 200,
    ):
        """
        ItemTracker object constructor.

        Args:
            max_disappeared_frames (int): Maximum number of frames before deregister.
            guard (int): When packet depth is cropped, the resulting crop will have 'guard' extra pixels on each side.
            max_item_distance (int): Maximal distance which packet can travel when it disappears in frame pixels.
        """

        self.tracked_item_list = []
        self.next_item_id = 0

        self.max_item_distance = max_item_distance
        self.max_disappeared_frames = max_disappeared_frames
        self.guard = guard

    def register_item(self, item: Packet):
        """
        Adds new item into database.

        Args:
            item (Packet): New packet object which should be added to the database.
        """

        item.id = self.next_item_id
        item.disappeared = 0
        self.next_item_id += 1
        self.tracked_item_list.append(item)

    def deregister_item(self, id: int):
        """
        Removes item with matching id from tracking database.

        Args:
            id (int): New item object whose parameters are transferred.
        """

        for tracked_item_index, tracked_item in enumerate(self.tracked_item_list):
            if tracked_item.id == id:
                del self.tracked_item_list[tracked_item_index]
                break

    def update_item(
        self,
        new_item: Packet,
        tracked_item: Packet,
        encoder_pos: int,
    ) -> Packet:
        """
        Updates parameters of single tracked packet with those of a new packet.

        Args:
            new_item (Packet): New packet object whose parameters are transferred.
            tracked_item (Packet): Packet object whose parameters are updated.
            encoder_pos (float): Position of encoder.

        Returns:
            tracked_item (Packet): Updated tracked packet object.
        """

        if tracked_item.id != new_item.id:
            print("[WARNING] Tried to update two items with different IDs together")
            return

        tracked_item.centroid_px = new_item.centroid_px
        tracked_item.type = new_item.type
        tracked_item.homography_matrix = new_item.homography_matrix
        tracked_item.bounding_width_px = new_item.bounding_width_px
        tracked_item.bounding_height_px = new_item.bounding_height_px
        tracked_item.add_angle_to_average(new_item.avg_angle_deg)
        tracked_item.set_base_encoder_position(encoder_pos)
        tracked_item.disappeared = 0

        # OBSOLETE parameters
        tracked_item.width = new_item.width
        tracked_item.height = new_item.height

        return tracked_item

    def update_tracked_item_list(
        self,
        labeled_item_list: list[Packet],
        encoder_pos: int,
    ):
        """
        Update tracked item database using labeled detected items.
        When item has id of None, it is registered as new item.
        When item has same id already tracked item, the tracked item is updated with detected parameters.

        Args:
            labeled_item_list (list[Packet]): List of detected Packet objects with id == id of nearest tracked item.
            encoder_pos (float): Position of encoder.
        """

        # Increment disappeared frame on all items
        for tracked_item in self.tracked_item_list:
            tracked_item.disappeared += 1

        for labeled_item in labeled_item_list:
            # Register new item
            if labeled_item.id == None:
                self.register_item(labeled_item)
                continue

            # Update exitsing item data
            for tracked_item_index, tracked_item in enumerate(self.tracked_item_list):
                if labeled_item.id == tracked_item.id:
                    self.tracked_item_list[tracked_item_index] = self.update_item(
                        labeled_item, tracked_item, encoder_pos
                    )
                    break

        # Check for items ready to be deregistered
        for tracked_item in self.tracked_item_list:
            if tracked_item.disappeared > self.max_disappeared_frames and self.max_disappeared_frames >= 1:
                self.deregister_item(tracked_item.id)

    def track_items(self, detected_item_list: list[Packet]) -> list[Packet]:
        """
        Labels input items with IDs from tracked item database, depending on distance.

        Args:
            detected_item_list (list[Packet]): List of detected Item objects with id == None.

        Returns:
            labeled_item_list (list[Packet]): List of detected Item objects with id == id of nearest tracked item.
        """

        labeled_item_list = detected_item_list
        # If no packets are being detected or tracked
        if len(detected_item_list) == 0 or len(self.tracked_item_list) == 0:
            return labeled_item_list
        else:
            # Create a list of tracked and detected centroids
            trackedCentroids = [item.centroid_px for item in self.tracked_item_list]
            detectCentroids = [item.centroid_px for item in detected_item_list]
            # Compute the distance between each pair of items
            distances = dist.cdist(
                np.array(trackedCentroids), np.array(detectCentroids)
            )
            # Sort tracked items (rows) by minimal distance
            tracked = distances.min(axis=1).argsort()
            # Sort detected items (columns) by minimal distance
            detected = distances.argmin(axis=1)[tracked]

            usedTracked = set()
            usedDetected = set()
            # Loop over the combination of the (row, column) index, starting from minimal distances
            for (trac, det) in zip(tracked, detected):
                # Ignore already used items
                if trac in usedTracked or det in usedDetected:
                    continue
                # If assigned distance is too far, ignore it
                if distances[trac, det] > self.max_item_distance:
                    continue
                # Assign id to detected item
                labeled_item_list[det].id = self.tracked_item_list[trac].id

                # Indicate which items were used
                usedTracked.add(trac)
                usedDetected.add(det)

        return labeled_item_list
