import numpy as np
from torch.utils.data import Dataset

from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize

import lidar_det.utils.jrdb_transforms as jt
import lidar_det.utils.utils_box3d as ub3d

from ._jrdb_handle import JRDBHandleDet3D
from .utils import get_prediction_target, collate_sparse_tensors

__all__ = [
    "JRDBDet3D",
]

_AVERAGE_BOX_LWH = (0.9, 0.5, 1.7)


_JRDB_VAL_SEQUENCES = [
    "clark-center-2019-02-28_1",
    "gates-ai-lab-2019-02-08_0",
    "huang-2-2019-01-25_0",
    "meyer-green-2019-03-16_0",
    "nvidia-aud-2019-04-18_0",
    "tressider-2019-03-16_1",
    "tressider-2019-04-26_2",
]


class JRDBDet3D(Dataset):
    def __init__(self, data_dir, split, cfg):
        if split == "train":
            self.__handle = JRDBHandleDet3D(
                data_dir, "train", exclude_sequences=_JRDB_VAL_SEQUENCES
            )
        elif split == "val":
            self.__handle = JRDBHandleDet3D(
                data_dir, "train", sequences=_JRDB_VAL_SEQUENCES
            )
        elif split == "train_val":
            self.__handle = JRDBHandleDet3D(data_dir, "train")

        elif split == "test":
            self.__handle = JRDBHandleDet3D(data_dir, "test")
        else:
            raise RuntimeError(f"Invalid split: {split}")

        self.__split = split
        self._augmentation = cfg["augmentation"] and "train" in split
        self._voxel_size = cfg["voxel_size"]
        self._num_points = cfg["num_points"]
        self._na = cfg["num_anchors"]
        self._no = cfg["num_ori_bins"]
        self._canonical = cfg["canonical"]
        self._target_mode = cfg["target_mode"]
        self._dist_min = cfg["dist_min"]
        self._dist_max = cfg["dist_max"]

    @property
    def split(self):
        return self.__split  # used by trainer.py

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        if self.__split == "test":
            return self.getitem_test_set(idx)

        data_dict = self.__handle[idx]

        # point cloud in base frame
        pc_upper = data_dict["pc_upper"]
        pc_lower = data_dict["pc_lower"]
        pc_upper = jt.transform_pts_upper_velodyne_to_base(pc_upper)
        pc_lower = jt.transform_pts_lower_velodyne_to_base(pc_lower)
        pc = np.concatenate([pc_upper, pc_lower], axis=1)  # (3, N)

        # bounding box in base frame
        boxes, _ = ub3d.string_to_boxes(data_dict["label_str"])

        # filter out corrupted annotations with negative dimension
        valid_mask = (boxes[:, 3:6] > 0.0).min(axis=1).astype(np.bool)
        boxes = boxes[valid_mask]

        # augmentation
        if self._augmentation:
            # random scale
            scale_factor = np.random.uniform(0.95, 1.05)
            pc *= scale_factor
            boxes[:, :6] *= scale_factor

            # random rotation
            theta = np.random.uniform(0, 2 * np.pi)
            rot_mat = np.array(
                [
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            pc = rot_mat @ pc
            boxes[:, :3] = boxes[:, :3] @ rot_mat.T
            boxes[:, 6] += theta

        # get regression label by assigning bounding box to point cloud
        pred_target, target_box_inds, target_boxes = get_prediction_target(
            pc,
            boxes,
            _AVERAGE_BOX_LWH,
            num_anchors=self._na,
            num_theta_bins=self._no,
            canonical=self._canonical,
            target_mode=self._target_mode,
            dist_min=self._dist_min,
            dist_max=self._dist_max,
        )  # (N, NA, C)

        # to voxel
        pc_voxel = np.round(pc / self._voxel_size)
        pc_voxel -= pc_voxel.min(axis=1, keepdims=True)
        pc_voxel = pc_voxel.T

        # NOTE all this does is find indices of non-duplicating elements
        inds, inverse_map = sparse_quantize(
            pc_voxel, feats=pc.T, labels=None, return_index=True, return_invs=True,
        )

        # # # TODO is this needed?
        # # if "train" in self.split:
        # #     if len(inds) > self.num_points:
        # #         inds = np.random.choice(inds, self.num_points, replace=False)

        net_input = SparseTensor(pc.T[inds], pc_voxel[inds])
        net_target = SparseTensor(
            pred_target.reshape(pc.shape[1], -1)[inds], pc_voxel[inds]
        )
        net_target_boxes = SparseTensor(target_boxes[inds], pc_voxel[inds])
        # target_full = SparseTensor(reg_labels.T, pc_voxel)
        # inverse_map = SparseTensor(inverse_map, pc_voxel)

        data_dict.update(
            {
                "net_input": net_input,
                "net_target": net_target,
                "net_target_boxes": net_target_boxes,
                # "target_full": target_full,
                "inverse_map": inverse_map,
                "points": pc,  # (3, N)
                "boxes": boxes,  # (B, 7)
                "pred_target": pred_target,  # (N, NA, 8)
                "target_box_inds": target_box_inds,  # (N,)
                "target_boxes": target_boxes,  # (N,)
                "ave_lwh": _AVERAGE_BOX_LWH,
                "canonical": self._canonical,
            }
        )

        return data_dict

    def getitem_test_set(self, idx):
        data_dict = self.__handle[idx]

        # point cloud in base frame
        pc_upper = data_dict["pc_upper"]
        pc_lower = data_dict["pc_lower"]
        pc_upper = jt.transform_pts_upper_velodyne_to_base(pc_upper)
        pc_lower = jt.transform_pts_lower_velodyne_to_base(pc_lower)
        pc = np.concatenate([pc_upper, pc_lower], axis=1)  # (3, N)

        # to voxel
        pc_voxel = np.round(pc / self._voxel_size)
        pc_voxel -= pc_voxel.min(axis=1, keepdims=True)
        pc_voxel = pc_voxel.T

        # NOTE all this does is find indices of non-duplicating elements
        inds, inverse_map = sparse_quantize(
            pc_voxel, feats=pc.T, labels=None, return_index=True, return_invs=True,
        )

        # # # TODO is this needed?
        # # if "train" in self.split:
        # #     if len(inds) > self.num_points:
        # #         inds = np.random.choice(inds, self.num_points, replace=False)

        net_input = SparseTensor(pc.T[inds], pc_voxel[inds])
        # inverse_map = SparseTensor(inverse_map, pc_voxel)

        data_dict.update(
            {
                "net_input": net_input,
                "inverse_map": inverse_map,
                "points": pc,  # (3, N)
                "ave_lwh": _AVERAGE_BOX_LWH,
                "canonical": self._canonical,
            }
        )

        return data_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, v in batch[0].items():
            if isinstance(v, SparseTensor):
                rtn_dict[k] = collate_sparse_tensors([sample[k] for sample in batch])
            elif k in ("ave_lwh", "canonical"):
                rtn_dict[k] = v
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict
