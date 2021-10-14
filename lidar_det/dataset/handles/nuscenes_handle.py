import numpy as np
import os

from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nuscenes_splits
from nuscenes.utils.data_classes import LidarPointCloud


__all__ = ["NuScenesHandle"]


class NuScenesHandle:
    def __init__(self, data_dir, split, mini=False, nsweeps=1):
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.data_dir = data_dir

        if mini:
            nusc_version = "v1.0-mini"
            split_scenes = (
                nuscenes_splits.mini_train
                if split == "train"
                else nuscenes_splits.mini_val
            )
        elif split == "test":
            nusc_version = "v1.0-test"
            split_scenes = nuscenes_splits.test
        else:
            nusc_version = "v1.0-trainval"
            if split == "train":
                split_scenes = nuscenes_splits.train
            elif split == "val":
                split_scenes = nuscenes_splits.val
            elif split == "train_val":
                split_scenes = nuscenes_splits.train + nuscenes_splits.val
            else:
                raise RuntimeError(f"Invalid split: {split}")

        self._nusc = NuScenes(version=nusc_version, dataroot=data_dir, verbose=False)
        self._nsweeps = nsweeps

        # locate scenes for the split
        all_scenes = self._nusc.scene
        all_scene_names = [s["name"] for s in all_scenes]

        scenes = [all_scenes[all_scene_names.index(s)] for s in split_scenes]
        self._scenes = scenes

        # build a flat list of all samples
        self.__flat_sample_tokens = []
        for scene_idx, scene in enumerate(scenes):
            s_t = scene["first_sample_token"]
            while s_t != "":
                self.__flat_sample_tokens.append(s_t)
                s_t = self._nusc.get("sample", s_t)["next"]

    def __len__(self):
        return len(self.__flat_sample_tokens)

    def __getitem__(self, idx):
        s_t = self.__flat_sample_tokens[idx]
        s_dict = self._nusc.get("sample", s_t)

        pc, dts = self.load_pointcloud(s_dict, self._nsweeps)

        anns = [self._nusc.get("sample_annotation", a_t) for a_t in s_dict["anns"]]

        frame_dict = {
            "frame_id": idx,
            "scene_token": s_dict["scene_token"],
            "sample_token": s_t,
            "first_sample": s_dict["prev"] == "",
            "pc": pc,
            "anns": anns,
        }

        if dts is not None:
            frame_dict["pc_dt"] = dts

        return frame_dict

    def load_pointcloud(self, sample_dict, nsweeps):
        """Load a point cloud given nuScenes sample dict of the point cloud

        Returns:
            pc (np.ndarray[3, N]): Point cloud in global frame
        """
        pc_dict = self._nusc.get("sample_data", sample_dict["data"]["LIDAR_TOP"])
        if nsweeps == 1:
            url = os.path.join(self.data_dir, pc_dict["filename"])
            pc = LidarPointCloud.from_file(url)
            dts = None
        else:
            # Example https://github.com/nutonomy/nuscenes-devkit/blob/5325d1b400950f777cd701bdd5e30a9d57d2eaa8/python-sdk/nuscenes/nuscenes.py#L1155  # noqa
            pc, dts = LidarPointCloud.from_file_multisweep(
                self._nusc, sample_dict, "LIDAR_TOP", "LIDAR_TOP", nsweeps=nsweeps
            )
            dts = dts[0]

        # Transform to global frame
        # From https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L749  # noqa
        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.  # noqa
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.  # noqa
        cs_record = self._nusc.get(
            "calibrated_sensor", pc_dict["calibrated_sensor_token"]
        )
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_record["translation"]))

        # Second step: transform from ego to the global frame.
        poserecord = self._nusc.get("ego_pose", pc_dict["ego_pose_token"])
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc.translate(np.array(poserecord["translation"]))

        return pc, dts
