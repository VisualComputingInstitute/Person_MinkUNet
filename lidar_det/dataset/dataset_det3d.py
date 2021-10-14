import numpy as np
import torch
from torch.utils.data import Dataset

from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize

import lidar_det.utils.jrdb_transforms as jt
import lidar_det.utils.utils_box3d as ub3d

from .utils import collate_sparse_tensors, boxes_to_target

# from .utils import get_prediction_target

__all__ = [
    "JRDBDet3D",
    "NuScenesDet3D",
]


class _DatasetBase(Dataset):
    def __init__(self, data_dir, split, cfg):
        vs = cfg["voxel_size"]
        voxel_size = (
            np.array(vs, dtype=np.float32)
            if isinstance(vs, list)
            else np.array([vs, vs, vs], dtype=np.float32)
        )
        self._voxel_size = voxel_size.reshape(3, 1)
        self._voxel_offset = np.array([1e5, 1e5, 1e4], dtype=np.int32).reshape(3, 1)
        self._num_points = cfg["num_points"]
        self._na = cfg["num_anchors"]
        self._no = cfg["num_ori_bins"]
        self._canonical = cfg["canonical"]
        self._included_classes = cfg["included_classes"]
        self._additional_features = cfg["additional_features"]
        self._nsweeps = cfg["nsweeps"]
        self._augmentation = cfg["augmentation"]

        self.__training = "train" in split  # loss will be computed
        self.__split = split
        self.__handle = self._get_handle(data_dir, split)

    def _get_handle(self, data_dir, split):
        raise NotImplementedError

    def _get_data(self, data_dict, training=True):
        raise NotImplementedError

    def _do_augmentation(self, pc, boxes):
        # random scale
        scale_factor = np.random.uniform(0.95, 1.05)
        pc *= scale_factor

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

        if boxes is not None and len(boxes) > 0:
            boxes[:, :6] *= scale_factor
            boxes[:, :3] = boxes[:, :3] @ rot_mat.T
            boxes[:, 6] += theta

        return pc, boxes

    @property
    def split(self):
        return self.__split  # used by trainer.py

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        data_dict = self.__handle[idx]

        pc, boxes_gt, boxes_gt_cls, pc_offset, addi_feats = self._get_data(data_dict)

        if self.__training and self._augmentation:
            pc, boxes_gt = self._do_augmentation(pc, boxes_gt)

        # voxel coordinate
        pc_voxel = np.round(pc / self._voxel_size) + self._voxel_offset
        pc_voxel = pc_voxel.T
        inds, inverse_map = sparse_quantize(
            pc_voxel, feats=None, labels=None, return_index=True, return_invs=True,
        )  # NOTE all this does is find indices of non-duplicating elements

        # for nuScenes with multisweep, only do prediction for keyframe voxels
        if "pc_dt" in data_dict:
            pc_dt = data_dict["pc_dt"]
            pc_kfmask = pc_dt == pc_dt.min()
            net_input_kfmask = pc_kfmask[inds]
            net_input_kfmask[inverse_map[pc_kfmask]] = 1
            # print("pc_kfmask", pc_kfmask.shape, pc_kfmask.sum())
            # print("net_input_kfmask", net_input_kfmask.shape, net_input_kfmask.sum())
        else:
            pc_kfmask = None
            net_input_kfmask = None

        # upper cap on memory consumption
        if self.__training and len(inds) > self._num_points:
            kept_inds = np.random.choice(len(inds), self._num_points, replace=False)
            inds = inds[kept_inds]
            if net_input_kfmask is not None:
                net_input_kfmask = net_input_kfmask[kept_inds]

        input_feat = (
            pc.T[inds]
            if addi_feats is None
            else np.concatenate((pc.T[inds], addi_feats.T[inds]), axis=1)
        )  # (N, C)
        net_input = SparseTensor(input_feat, pc_voxel[inds])
        if net_input_kfmask is not None:
            net_input_kfmask = torch.from_numpy(net_input_kfmask).bool()

        params_dict = {
            "ave_lwh": self._ave_lwh,
            "canonical": self._canonical,
            "voxel_offset": self._voxel_offset,
            "voxel_size": self._voxel_size,
            "class_mapping": self._inds_to_cls,
            "dist_thresh": self._dist_thresh,
        }

        data_dict.update(
            {
                "net_input": net_input,
                "net_input_kfmask": net_input_kfmask,
                # "inverse_map": inverse_map,
                "points": pc,  # (3, N)
                "points_offset": pc_offset,  # (3,)
                "points_kfmask": pc_kfmask,  # (N, )
                "num_voxels": len(inds),
                "additional_features": addi_feats,  # (C, N) or None
                "boxes_gt": boxes_gt,  # (B, 7) or None
                "boxes_gt_cls": boxes_gt_cls,  # (B,) or None
                "params": params_dict,
            }
        )

        if not self.__training:
            return data_dict

        # assigning target for each class independently
        N = len(inds)
        A = self._na
        S = self._nc
        btmp = boxes_to_target(np.ones((1, 7)), self._ave_lwh[0], A, self._no)
        C = btmp.shape[-1]
        closest_box_inds = -1 * np.ones((N, S), dtype=np.int32)
        boxes_matched = np.zeros((N, S, 7), dtype=np.float32)
        boxes_encoded = np.zeros((N, A, S, C), dtype=np.float32)
        if boxes_gt is not None:
            for icls in range(self._nc):
                cmask = boxes_gt_cls == icls
                boxes_gt_c = boxes_gt[cmask]
                if len(boxes_gt_c) == 0:
                    continue
                closest_box_inds_c, _ = ub3d.find_closest_boxes(pc, boxes_gt_c)
                closest_box_inds_c = closest_box_inds_c[inds]
                boxes_matched_c = boxes_gt_c[closest_box_inds_c]
                closest_box_inds[:, icls] = closest_box_inds_c
                boxes_matched[:, icls, :] = boxes_matched_c
                boxes_encoded[:, :, icls, :] = boxes_to_target(
                    boxes_matched_c, self._ave_lwh[icls], A, self._no
                )

        boxes_matched = torch.from_numpy(boxes_matched)
        boxes_encoded = torch.from_numpy(boxes_encoded)
        # boxes_cls = (
        #     torch.from_numpy(boxes_gt_cls[closest_box_inds])
        #     if boxes_gt_cls is not None
        #     else None
        # )

        data_dict.update(
            {
                "boxes_matched": boxes_matched,  # (N, S, 7)
                "boxes_encoded": boxes_encoded,  # (N, A, S, C)
                # "boxes_cls": boxes_cls,  # (N,)
                "closest_box_inds": closest_box_inds,  # (N, S)
            }
        )

        return data_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, v in batch[0].items():
            if isinstance(v, SparseTensor):
                rtn_dict[k] = collate_sparse_tensors([sample[k] for sample in batch])
            elif isinstance(v, torch.Tensor):
                rtn_dict[k] = torch.cat([sample[k] for sample in batch], dim=0)
            elif k == "params":
                if k not in rtn_dict:
                    rtn_dict[k] = v
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict


class JRDBDet3D(_DatasetBase):
    def __init__(self, *args, **kwargs):
        super(JRDBDet3D, self).__init__(*args, **kwargs)
        self._ave_lwh = [(0.9, 0.5, 1.7)]
        self._dist_thresh = [(0.5, 0.7)]
        self._nc = 1
        self._inds_to_cls = ["pedestrian"]  # not used

    def _get_handle(self, data_dir, split):
        from .handles.jrdb_handle import JRDBHandleDet3D

        jrdb_val_seq = [
            "clark-center-2019-02-28_1",
            "gates-ai-lab-2019-02-08_0",
            "huang-2-2019-01-25_0",
            "meyer-green-2019-03-16_0",
            "nvidia-aud-2019-04-18_0",
            "tressider-2019-03-16_1",
            "tressider-2019-04-26_2",
        ]

        if split == "train":
            return JRDBHandleDet3D(data_dir, "train", exclude_sequences=jrdb_val_seq)
        elif split == "val":
            return JRDBHandleDet3D(data_dir, "train", sequences=jrdb_val_seq)
        elif split == "train_val":
            return JRDBHandleDet3D(data_dir, "train")
        elif split == "test":
            return JRDBHandleDet3D(data_dir, "test")
        else:
            raise RuntimeError(f"Invalid split: {split}")

    def _get_data(self, data_dict):
        # point cloud in base frame
        pc_upper = data_dict["pc_upper"]
        pc_lower = data_dict["pc_lower"]
        pc_upper = jt.transform_pts_upper_velodyne_to_base(pc_upper)
        pc_lower = jt.transform_pts_lower_velodyne_to_base(pc_lower)
        pc = np.concatenate([pc_upper, pc_lower], axis=1)  # (3, N)
        pc_offset = np.zeros(3, dtype=np.float32)

        if "label_str" not in data_dict.keys():
            return pc, None, None, pc_offset, None

        # bounding box in base frame
        boxes, _ = ub3d.string_to_boxes(data_dict["label_str"])

        # filter out corrupted annotations with negative dimension
        valid_mask = (boxes[:, 3:6] > 0.0).min(axis=1).astype(np.bool)
        boxes = boxes[valid_mask]
        boxes_cls = np.zeros(len(boxes), dtype=np.int32)

        return pc, boxes, boxes_cls, pc_offset, None


class NuScenesDet3D(_DatasetBase):
    def __init__(self, *args, **kwargs):
        super(NuScenesDet3D, self).__init__(*args, **kwargs)
        self._ave_lwh = [
            (0.50, 2.53, 0.98),
            (1.70, 0.60, 1.28),
            (11.23, 2.93, 3.47),
            (4.62, 1.95, 1.73),
            (6.37, 2.85, 3.19),
            (2.11, 0.77, 1.47),
            (0.73, 0.67, 1.77),
            (0.41, 0.41, 1.07),
            (12.29, 2.90, 3.87),
            (6.93, 2.51, 2.84),
        ]  # from nusc.list_category()
        self._dist_thresh = [
            (0.6, 2.63),
            (0.7, 1.8),
            (3.03, 11.33),
            (2.05, 4.72),
            (2.95, 6.47),
            (0.87, 2.21),
            (0.77, 0.83),
            (0.51, 0.71),
            (3.0, 12.39),
            (2.61, 7.03),
        ]
        self._nc = 10

        self._cls_mapping = {
            "animal": "void",
            "human.pedestrian.personal_mobility": "void",
            "human.pedestrian.stroller": "void",
            "human.pedestrian.wheelchair": "void",
            "movable_object.debris": "void",
            "movable_object.pushable_pullable": "void",
            "static_object.bicycle_rack": "void",
            "vehicle.emergency.ambulance": "void",
            "vehicle.emergency.police": "void",
            "movable_object.barrier": "barrier",
            "vehicle.bicycle": "bicycle",
            "vehicle.bus.bendy": "bus",
            "vehicle.bus.rigid": "bus",
            "vehicle.car": "car",
            "vehicle.construction": "construction_vehicle",
            "vehicle.motorcycle": "motorcycle",
            "human.pedestrian.adult": "pedestrian",
            "human.pedestrian.child": "pedestrian",
            "human.pedestrian.construction_worker": "pedestrian",
            "human.pedestrian.police_officer": "pedestrian",
            "movable_object.trafficcone": "traffic_cone",
            "vehicle.trailer": "trailer",
            "vehicle.truck": "truck",
        }

        self._cls_to_inds = {
            "void": -1,
            "barrier": 0,
            "bicycle": 1,
            "bus": 2,
            "car": 3,
            "construction_vehicle": 4,
            "motorcycle": 5,
            "pedestrian": 6,
            "traffic_cone": 7,
            "trailer": 8,
            "truck": 9,
        }

        self._inds_to_cls = [
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
        ]
        for i, c in enumerate(self._inds_to_cls):
            assert self._cls_to_inds[c] == i

        # customized classes
        nc = len(self._included_classes)
        if nc > 0:
            cls_to_inds = {"void": -1}
            inds_to_cls = []
            dist_thresh = []
            ave_lwh = []

            for i, c in enumerate(self._included_classes):
                cls_to_inds[c] = i
                inds_to_cls.append(c)
                idx = self._cls_to_inds[c]
                dist_thresh.append(self._dist_thresh[idx])
                ave_lwh.append(self._ave_lwh[idx])

            for k, c in self._cls_mapping.items():
                if c not in self._included_classes:
                    self._cls_mapping[k] = "void"

            self._nc = nc
            self._cls_to_inds = cls_to_inds
            self._inds_to_cls = inds_to_cls
            self._dist_thresh = dist_thresh
            self._ave_lwh = ave_lwh

    def _get_handle(self, data_dir, split):
        from .handles.nuscenes_handle import NuScenesHandle

        # return NuScenesHandle(data_dir, split, mini=True, nsweeps=self._nsweeps)
        return NuScenesHandle(data_dir, split, mini=False, nsweeps=self._nsweeps)

    def _get_data(self, data_dict):
        # point cloud in global frame
        pc = data_dict["pc"].points[:3]  # (3, N)

        # center point cloud
        pc_mean = pc.mean(axis=1, keepdims=True)
        pc -= pc_mean

        # additional features
        addi_feats = []
        if "intensity" in self._additional_features:
            intensity = (data_dict["pc"].points[3] / 255.0) - 0.5
            addi_feats.append(intensity)
        if "pc_dt" in data_dict and "time" in self._additional_features:
            addi_feats.append(data_dict["pc_dt"])
        addi_feats = np.stack(addi_feats, axis=0) if len(addi_feats) > 0 else None

        if len(data_dict["anns"]) == 0:
            return pc, None, None, pc_mean, addi_feats

        boxes = []
        boxes_cls = []
        for ann in data_dict["anns"]:
            cls_str = self._cls_mapping[ann["category_name"]]
            if cls_str != "void":
                box, _ = ub3d.box_from_nuscenes(ann)
                boxes.append(box)
                boxes_cls.append(self._cls_to_inds[cls_str])

        boxes = np.array(boxes, dtype=np.float32)
        boxes_cls = np.array(boxes_cls, dtype=np.int32)

        if boxes.shape[0] > 0:
            boxes[:, :3] = boxes[:, :3] - pc_mean.T

        return pc, boxes, boxes_cls, pc_mean, addi_feats
