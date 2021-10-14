import os
import cv2
import numpy as np

# Force the dataloader to load only one sample, in which case the network should
# fit perfectly.
_DEBUG_ONE_SAMPLE = False

__all__ = ["KITTIHandleDet3D"]


class KITTIHandleDet3D:
    def __init__(self, data_dir, split="train", frames=None):
        if _DEBUG_ONE_SAMPLE:
            split = "train"

        if split == "train" or self._split == "val":
            root_dir = os.path.join(data_dir, "object/training")
            self._label_dir = os.path.join(root_dir, "label_2")
        else:
            root_dir = os.path.join(data_dir, "object/testing")
            self._label_dir = None

        self._split = split
        self._image_dir = os.path.join(root_dir, "image_2")
        self._lidar_dir = os.path.join(root_dir, "velodyne")
        self._calib_dir = os.path.join(root_dir, "calib")
        self._plane_dir = os.path.join(root_dir, "planes")

        self.__frame_inds = (
            sorted([x.split(".")[0] for x in os.listdir(self._lidar_dir)])
            if frames is None
            else frames
        )

    def __len__(self):
        if _DEBUG_ONE_SAMPLE:
            return 80
        else:
            return len(self.__frame_inds)

    def __getitem__(self, idx):
        frame_id = self.__frame_inds[idx]

        pc = (
            np.fromfile(
                os.path.join(self._lidar_dir, f"{frame_id:06d}.bin"), dtype=np.float32
            )
            .reshape(-1, 4)
            .T
        )

        im = cv2.imread(os.path.join(self._image_dir, f"{frame_id:06d}.png"))

        frame_dict = {
            "frame_id": frame_id,
            "pc": pc,  # (4, N)
            "im": im,  # (H, W, 3)
            "calib_url": os.path.join(self._calib_dir, f"{frame_id:06d}.txt"),
        }

        if self._label_dir is not None:
            frame_dict["plane"] = self._get_road_plane(frame_id)
            frame_dict["label_url"] = (
                os.path.join(self._label_dir, f"{frame_id:06d}.txt"),
            )

        return frame_dict

    def _get_road_plane(self, idx):
        plane_file = os.path.join(self._plane_dir, f"{idx:06d}.txt")
        with open(plane_file, "r") as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm

        return plane
