import time

import torch
import numpy as np

from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize

from lidar_det.model import get_model
from lidar_det.dataset.utils import target_to_boxes_torch, collate_sparse_tensors
from lidar_det.utils.utils_box3d import nms_3d_dist_gpu


class PersonMinkUNet(object):
    def __init__(self, ckpt_file, gpu=True):
        """A warpper class for end-to-end inference.

        Args:
            ckpt_file (str): Path to checkpoint
            gpu (bool): True to use GPU. Defaults to True.
        """
        self._gpu = gpu

        model_cfg = {
            "type": "MinkUNet",
            "kwargs": {
                "cr": 1.0,
                "run_up": True,
                "num_anchors": 1,
                "num_ori_bins": 12,
                "fpn": False,
            },
            "target_mode": 2,
            "disentangled_loss": False,
        }

        self._net = get_model(model_cfg, inference_only=True)
        self._num_anchors = model_cfg["kwargs"]["num_anchors"]
        self._num_cls = 1  # person is the only class
        self._ave_lwh = (0.9, 0.5, 1.7)

        self._voxel_size = np.array([0.05, 0.05, 0.1], dtype=np.float32).reshape(3, 1)
        self._voxel_offset = np.array([1e5, 1e5, 1e4], dtype=np.int32).reshape(3, 1)

        self._voxel_size_torch = torch.from_numpy(self._voxel_size).float()
        self._voxel_offset_torch = torch.from_numpy(self._voxel_offset).float()

        ckpt = torch.load(ckpt_file)
        self._net.load_state_dict(ckpt["model_state"])

        self._net.eval()

        if gpu:
            # NOTE Set to False gives slightly faster speed
            torch.backends.cudnn.benchmark = False
            self._net = self._net.cuda()
            self._voxel_size_torch = self._voxel_size_torch.cuda(
                non_blocking=True
            ).float()
            self._voxel_offset_torch = self._voxel_offset_torch.cuda(
                non_blocking=True
            ).float()

        print(
            "Detector initialized\n",
            f"GPU: {gpu}\n",
            f"ckpt: {ckpt_file}\n",
            f"voxel size: {self._voxel_size}\n",
            f"voxel offset: {self._voxel_offset}\n",
        )

    def __call__(self, pc):
        """
        Args:
            pc (array[3, N]): Point cloud

        Returns:
            boxes (array[B, 7]): (x, y, z, l, w, h, rt)
            scores (array[B])
        """
        net_input = self._preprocess(pc)

        if self._gpu:
            net_input = net_input.cuda()

        # forward
        with torch.no_grad():
            net_pred = self._net(net_input)  # (N, C)
            boxes_nms, scores_nms = self._postprocess(net_pred, net_input)

        return boxes_nms, scores_nms

    def _preprocess(self, pc):
        pc_voxel = np.round(pc / self._voxel_size) + self._voxel_offset
        pc_voxel = pc_voxel.T

        # NOTE all this does is find indices of non-duplicating elements
        inds, inverse_map = sparse_quantize(
            pc_voxel, feats=pc.T, labels=None, return_index=True, return_invs=True,
        )
        net_input = SparseTensor(pc.T[inds], pc_voxel[inds])
        net_input = collate_sparse_tensors([net_input])

        return net_input

    def _postprocess(self, net_pred, net_input):
        # regression is w.r.t. to voxel center
        voxel_center = net_input.C[:, :3].clone().float() + 0.5
        voxel_center = (
            voxel_center - self._voxel_offset_torch.view(1, 3)
        ) * self._voxel_size_torch.view(1, 3)
        voxel_center = voxel_center[:, None, None, :]  # (N, 1, 3)

        # decode network prediction
        net_pred = net_pred.view(
            net_pred.shape[0], self._num_anchors, self._num_cls, -1
        )  # (N, A, S, C)
        cls_pred = torch.sigmoid(net_pred[..., 0])  # (N, A, S)
        reg_pred = net_pred[..., 1:]  # (N, A, S, C - 1)
        reg_pred[..., :3] = reg_pred[..., :3] + voxel_center  # offset from voxel center

        # postprocess prediction to get boxes
        num_theta_bins = int((reg_pred.shape[-1] - 6) / 2)
        boxes = target_to_boxes_torch(
            reg_pred[:, :, 0, :], self._ave_lwh, num_theta_bins
        ).view(
            -1, 7
        )  # (N * A, 7)

        # fast NMS based on distance
        cls_pred = cls_pred.view(-1)
        nms_inds = nms_3d_dist_gpu(
            boxes,
            cls_pred,
            l_ave=self._ave_lwh[0],
            w_ave=self._ave_lwh[1],
            nms_thresh=0.4,
        )
        boxes_nms = boxes[nms_inds].data.cpu().numpy()
        scores_nms = cls_pred[nms_inds].data.cpu().numpy()  # in descending order

        return boxes_nms, scores_nms


class DetectorWithClock(PersonMinkUNet):
    def __init__(self, *args, **kwargs):
        """For measuring inference speed"""
        super(DetectorWithClock, self).__init__(*args, **kwargs)

        self._pc_count = []
        self._voxel_count = []
        self._time_preprocess = []
        self._time_forward = []
        self._time_postprocess = []

    def __call__(self, pc):
        """
        Args:
            pc (array[3, N]): Point cloud

        Returns:
            boxes (array[B, 7]): (x, y, z, l, w, h, rt)
            scores (array[B])
        """
        t0 = time.time()

        net_input = self._preprocess(pc)

        if self._gpu:
            net_input = net_input.cuda()

        torch.cuda.synchronize()
        t1 = time.time()

        # forward
        with torch.no_grad():
            net_pred = self._net(net_input)  # (N, C)
            torch.cuda.synchronize()
            t2 = time.time()

            boxes_nms, scores_nms = self._postprocess(net_pred, net_input)
            torch.cuda.synchronize()
            t3 = time.time()

        self._time_preprocess.append(t1 - t0)
        self._time_forward.append(t2 - t1)
        self._time_postprocess.append(t3 - t2)
        self._pc_count.append(pc.shape[1])
        self._voxel_count.append(net_input.F.shape[0])

        return boxes_nms, scores_nms

    def __str__(self):
        def val2stat(val_list):
            arr = np.array(val_list)
            return f"{arr.mean():f} ({arr.std():f})"

        fps = 1.0 / (
            np.array(self._time_forward)
            + np.array(self._time_preprocess)
            + np.array(self._time_postprocess)
        )

        s = (
            "Summary [ave (std)]\n"
            f"Frame count: {len(self._time_preprocess)}\n"
            f"Preprocess time: {val2stat(self._time_preprocess)}\n"
            f"Forward time: {val2stat(self._time_forward)}\n"
            f"Postprocess time: {val2stat(self._time_postprocess)}\n"
            f"FPS: {val2stat(fps)}\n"
            f"Point per frame: {val2stat(self._pc_count)}\n"
            f"Voxel per frame: {val2stat(self._voxel_count)}\n"
        )
        return s

    def get_time(self):
        return (
            self._time_preprocess,
            self._time_forward,
            self._time_postprocess,
            self._pc_count,
            self._voxel_count,
        )

    def reset(self):
        self._pc_count = []
        self._voxel_count = []
        self._time_preprocess = []
        self._time_forward = []
        self._time_postprocess = []


if __name__ == "__main__":
    from lidar_det.dataset.handles.jrdb_handle import JRDBHandleDet3D
    import lidar_det.utils.jrdb_transforms as tu
    from lidar_det.utils.utils_box3d import boxes_to_corners

    from mayavi import mlab

    # some color
    gs_blue = (66.0 / 256, 133.0 / 256, 244.0 / 256)
    gs_red = (234.0 / 256, 68.0 / 256, 52.0 / 256)
    gs_yellow = (251.0 / 256, 188.0 / 256, 4.0 / 256)
    gs_green = (52.0 / 256, 168.0 / 256, 83.0 / 256)
    gs_orange = (255.0 / 256, 109.0 / 256, 1.0 / 256)
    gs_blue_light = (70.0 / 256, 189.0 / 256, 196.0 / 256)

    ckpt_path = "/globalwork/jia/share/JRDB_cvpr21_workshop/logs/unet_bl_voxel_jrdb_0.05_0.1_20210519_232859/ckpt/ckpt_e40.pth"  # noqa
    detector = DetectorWithClock(ckpt_path)

    loader = JRDBHandleDet3D(data_dir="./data/JRDB", split="train")

    for data_dict in loader:
        pc_xyz_upper = tu.transform_pts_upper_velodyne_to_base(data_dict["pc_upper"])
        pc_xyz_lower = tu.transform_pts_lower_velodyne_to_base(data_dict["pc_lower"])
        pc = np.concatenate((pc_xyz_upper, pc_xyz_lower), axis=1)

        boxes, scores = detector(pc)
        boxes = boxes[scores > 0.5]
        print(boxes.shape)

        fig = mlab.figure(
            figure=None,
            bgcolor=(1, 1, 1),
            fgcolor=(0, 0, 0),
            engine=None,
            size=(1600, 1000),
        )

        mlab.points3d(
            pc[0], pc[1], pc[2], scale_factor=0.05, color=gs_blue, figure=fig,
        )

        corners_xyz, connect_inds = boxes_to_corners(boxes, connect_inds=True)
        for corner_xyz in corners_xyz:
            for inds in connect_inds:
                mlab.plot3d(
                    corner_xyz[0, inds],
                    corner_xyz[1, inds],
                    corner_xyz[2, inds],
                    tube_radius=None,
                    line_width=3,
                    color=gs_yellow,
                    figure=fig,
                )

        mlab.view(focalpoint=(0, 0, 0))
        mlab.move(190, 0, 5.0)
        mlab.pitch(-10)

        mlab.show()
        break

    print(detector)
