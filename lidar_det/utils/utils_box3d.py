import numpy as np

# from numba import jit
import torch

from shapely.geometry import Polygon
from pyquaternion import Quaternion

import iou3d


def _hypot(x, y):
    # TODO use torch.hypot for PyTorch >= 1.7
    return torch.sqrt(torch.square(x) + torch.square(y))


"""
Fast (vectorized) boxes API.

Boxes should be numpy.array[B, 7] with (x, y, z, l, w, h, theta)

Boxes follow JRDB base frame convention, x-forward, y-left, z-up, theta is the
angle between x-axis and the length side of bounding boxes
"""


def get_R(boxes):
    """Get rotation matrix R along z axis that transforms a point from box coordinate
    to world coordinate.

    Args:
        boxes (array[B, 7]): (x, y, z, l, w, h, theta) of each box

    Returns:
        Rs (array[B, 3, 3])
    """
    # NOTE plus pi specifically for JRDB, don't know the reason
    theta = boxes[:, 6] + np.pi
    cs, ss = np.cos(theta), np.sin(theta)
    zeros, ones = np.zeros(len(cs)), np.ones(len(cs))
    Rs = np.array(
        [[cs, ss, zeros], [-ss, cs, zeros], [zeros, zeros, ones]], dtype=np.float32
    )  # (3, 3, B)

    return Rs.transpose((2, 0, 1))


def get_R_torch(boxes):
    """Get rotation matrix R along z axis that transforms a point from box coordinate
    to world coordinate.

    Args:
        boxes (tensor[B, 7]): (x, y, z, l, w, h, theta) of each box

    Returns:
        Rs (tensor[B, 3, 3])
    """
    # NOTE plus pi specifically for JRDB, don't know the reason
    theta = boxes[:, 6] + np.pi
    cs, ss = torch.cos(theta), torch.sin(theta)
    Rs = torch.zeros((len(cs), 3, 3), device=boxes.device).float()
    Rs[:, 2, 2] = 1
    Rs[:, 0, 0] = cs
    Rs[:, 0, 1] = ss
    Rs[:, 1, 0] = -ss
    Rs[:, 1, 1] = cs

    return Rs


def boxes_to_corners(boxes, resize_factor=1.0, connect_inds=False):
    """Return xyz coordinates of the eight vertices of the bounding box

    First four points are fl (front left), fr, br, bl on top plane. Last four
    points are same order, but for the bottom plane.

          0 -------- 1        __
         /|         /|        //|
        3 -------- 2 .       //
        | |        | |      front
        . 4 -------- 5
        |/         |/
        7 -------- 6

    To draw a box, do something like

    corners, connect_inds = boxes_to_corners(boxes)
    for corner in corners:
        for inds in connect_inds:
            mlat.plot3d(corner[0, inds], corner[1, inds], corner[2, inds],
                        tube_radius=None, line_width=5)

    Args:
        boxes (array[B, 7]): (x, y, z, l, w, h, theta) of each box
        resize_factor (float): resize box lwh dimension
        connect_inds(bool): true will also return a list of indices for drawing
            the box as line segments

    Returns:
        corners_xyz (array[B, 3, 8])
        connect_inds (tuple[list[int]])
    """
    # in box frame
    c_xyz = np.array(
        [
            [1, 1, -1, -1, 1, 1, -1, -1],
            [-1, 1, 1, -1, -1, 1, 1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
        ],
        dtype=np.float32,
    )  # (3, 8)
    c_xyz = 0.5 * c_xyz[np.newaxis, :, :] * boxes[:, 3:6, np.newaxis]  # (B, 3, 8)
    c_xyz = c_xyz * resize_factor

    # to world frame
    R = get_R(boxes)  # (B, 3, 3)
    c_xyz = R @ c_xyz + boxes[:, :3, np.newaxis]  # (B, 3, 8)

    if not connect_inds:
        return c_xyz
    else:
        l1 = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 1]
        l2 = [0, 5, 1]
        l3 = [2, 6]
        l4 = [3, 7]
        return c_xyz, (l1, l2, l3, l4)


def boxes_to_corners_torch(boxes):
    """See boxes_to_corners()

    Args:
        boxes (tensor[B, 7]): (x, y, z, l, w, h, theta) of each box

    Returns:
        corners_xyz (tensor[B, 3, 8])
    """
    # in box frame
    c_xyz = torch.tensor(
        [
            [1, 1, -1, -1, 1, 1, -1, -1],
            [-1, 1, 1, -1, -1, 1, 1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
        ],
        device=boxes.device,
    ).float()  # (3, 8)
    c_xyz = 0.5 * c_xyz.unsqueeze(dim=0) * boxes[:, 3:6].unsqueeze(dim=2)  # (B, 3, 8)

    # to world frame
    R = get_R_torch(boxes)  # (B, 3, 3)
    c_xyz = R @ c_xyz + boxes[:, :3].unsqueeze(dim=2)  # (B, 3, 8)

    return c_xyz


def boxes_to_central_line(boxes):
    """Return parameter a, b, c such that ax + by + c = 0 is the line that passes
    the box center and is in parallel with the long edge in BEV.

    Args:
        boxes (array[..., 7]): (x, y, z, l, w, h, theta) of each box

    Returns:
        boxes_lp (array[..., 3]): (a, b, c) line parameters of each box
    """
    # in case length is shorter than width
    bmask = boxes[..., 3] < boxes[..., 4]
    theta = -boxes[..., 6]  # not sure why minus is needed
    theta[bmask] -= 0.5 * np.pi

    a = np.tan(theta)
    b = -np.ones_like(a)
    c = -a * boxes[..., 0] - b * boxes[..., 1]
    boxes_lp = np.stack((a, b, c), axis=-1)
    boxes_lp /= np.linalg.norm(boxes_lp, axis=-1, keepdims=True)

    return boxes_lp


def boxes_to_central_line_torch(boxes):
    """See boxes_to_central_line

    Args:
        boxes (tensor[..., 7]): (x, y, z, l, w, h, theta) of each box

    Returns:
        boxes_lp (tensor[..., 3]): (a, b, c) line parameters of each box
    """
    # in case length is shorter than width
    bmask = boxes[..., 3] < boxes[..., 4]
    theta = -boxes[..., 6]  # not sure why minus is needed
    theta[bmask] -= 0.5 * np.pi

    a = torch.tan(theta)
    b = -torch.ones_like(a)
    c = -a * boxes[..., 0] - b * boxes[..., 1]
    boxes_lp = torch.stack((a, b, c), dim=-1)
    boxes_lp /= torch.linalg.norm(boxes_lp, dim=-1, keepdim=True)

    return boxes_lp


def distance_pc_to_boxes_torch(
    pc, boxes, normalize=False, delta_size=0.0, return_in_box_mask=False
):
    """Compute distance between points and matching boxes

    Args:
        pc (tensor[..., 3]): Points
        boxes (tensor[..., 7]): (x, y, z, l, w, h, theta)

    Returns:
        dist (tensor[...])
    """
    # vector from points to boxes, in world frame
    xyz_offs = pc - boxes[..., :3]
    xyz_dist = torch.norm(xyz_offs, dim=-1)  # (...)

    if normalize:
        boxes_diag = 0.5 * torch.norm(boxes[..., 3:6] + delta_size, dim=-1)
        xyz_dist = xyz_dist / boxes_diag

    if not return_in_box_mask:
        return xyz_dist

    # convert offsets to box frame
    out_shape = boxes.shape[:-1]
    R = get_R_torch(boxes.view(-1, 7))  # (B, 3, 3), box to world
    xyz_offs = R.transpose(1, 2) @ xyz_offs.view(-1, 3, 1)  # (B, 3, 1)
    xyz_offs = xyz_offs.view(*out_shape, 3)  # (..., 3)

    hf_lwh = 0.5 * (boxes[..., 3:6] + delta_size)
    ib_mask = torch.logical_and(xyz_offs >= -hf_lwh, xyz_offs <= hf_lwh)
    ib_mask = ib_mask.min(dim=-1)[0]

    return xyz_dist, ib_mask


# @jit(nopython=True)
# def find_closest_boxes(pc, boxes):
#     """For each point, return the index of the box with closest center and the distance

#     Args:
#         pc (array[3, N]): xyz point cloud
#         boxes (array[B, 7]): bounding boxes (x, y, z, l, w, h, theta)

#     Returns:
#         closest_box_inds (array[N,]): The index of the closest box from a point
#         closest_box_dist (array[N,]): Distance to the box center
#     """
#     # offset from box center to points
#     N = pc.shape[1]
#     closest_box_inds = np.empty(N, dtype=np.int32)
#     closest_box_dist = np.empty(N, dtype=np.float32)
#     for n in range(N):
#         pc_n = pc[:, n : n + 1].T
#         pb_dist = pc_n - boxes[:, :3]  # (B, 3)
#         pb_norm_sq = np.square(pb_dist).sum(axis=1)  # (B, )

#         min_idx = pb_norm_sq.argmin()
#         closest_box_inds[n] = min_idx
#         closest_box_dist[n] = np.sqrt(pb_norm_sq[min_idx])

#     return closest_box_inds, closest_box_dist


def find_closest_boxes(pc, boxes):
    """For each point, return the index of the box with closest center and the distance

    Args:
        pc (array[3, N]): xyz point cloud
        boxes (array[B, 7]): bounding boxes (x, y, z, l, w, h, theta)

    Returns:
        closest_box_inds (array[N,]): The index of the closest box from a point
        closest_box_dist (array[N,]): Distance to the box center
    """
    # offset from box center to points
    pc_box_dist = pc[np.newaxis, :, :] - boxes[:, :3, np.newaxis]  # (B, 3, N)
    pc_box_norm = np.linalg.norm(pc_box_dist, axis=1)  # (B, N)
    closest_box_inds = pc_box_norm.argmin(axis=0)  # (N, )
    closest_box_dist = np.take_along_axis(
        pc_box_norm, closest_box_inds[np.newaxis, ...], axis=0
    )[0]

    return closest_box_inds, closest_box_dist


def find_in_box_points(pc, boxes, resize_factor=1.0):
    """For each point, return the indices of the box it belongs to. Only work
    with boxes that is rotated along z (verticle) axis.

    Args:
        pc (array[3, N]): xyz point cloud
        boxes (array[B, 7]): bounding boxes (x, y, z, l, w, h, theta)
        resize_factor (float): resize box dimension

    Returns:
        in_box_mask (array[N,]): True if a point is in at least one box
        closest_box_inds (array[N,]): The index of the closest box from a point
    """
    # offset from box center to points in world coordinate
    pc_box = pc[np.newaxis, :, :] - boxes[:, :3, np.newaxis]  # (B, 3, N)

    # transform the offset to box coordinate
    Rs = get_R(boxes)  # (B, 3, 3)
    pc_box = Rs.transpose((0, 2, 1)) @ pc_box  # (B, 3, N)

    # check if points in box
    lwh_half = 0.5 * boxes[:, 3:6, np.newaxis] * resize_factor  # (B, 3, 1)
    xyz_mask = np.logical_and(pc_box >= -lwh_half, pc_box <= lwh_half)  # (B, 3, N)
    xyz_mask = xyz_mask.min(axis=1)  # xyz all in range, (B, N)
    in_box_mask = xyz_mask.max(axis=0)  # belong to at least one box (N, )

    # in the case of a point within two boxes, it is assigned to the closer one
    pc_box_norm = np.linalg.norm(pc_box, axis=1)  # (B, N)
    closest_box_inds = pc_box_norm.argmin(axis=0)  # (N, )

    return in_box_mask, closest_box_inds


def get_iou3d(boxes, query_boxes, need_bev=False):
    """Compute pair-wise IoU between two group of boxes.

    Modified from
    https://github.com/sshaoshuai/PointRCNN/blob/master/lib/utils/kitti_utils.py#L195

    Args:
        boxes (array[N, 7]): (x, y, z, l, w, h, theta)
        query_boxes (array[M, 7])
        need_bev (bool): True to also return IoU of BEV

    Returns:
        iou_3d (array[N, M])
        iou_bev (array[N, M])
    """
    N, M = boxes.shape[0], query_boxes.shape[0]
    corners_a = boxes_to_corners(boxes)  # (N, 3, 8)
    corners_b = boxes_to_corners(query_boxes)

    iou_3d = np.zeros((N, M), dtype=np.float32)
    iou_bev = np.zeros((N, M), dtype=np.float32)

    min_z_a = corners_a[:, 2, -1]
    max_z_a = corners_a[:, 2, 0]
    min_z_b = corners_b[:, 2, -1]
    max_z_b = corners_b[:, 2, 0]

    for i in range(N):
        for j in range(M):
            max_of_min = np.max([min_z_a[i], min_z_b[j]])
            min_of_max = np.min([max_z_a[i], max_z_b[j]])
            z_overlap = np.max([0, min_of_max - max_of_min])
            if z_overlap == 0:
                continue

            bottom_a = Polygon(corners_a[i, :2, 4:].T)
            bottom_b = Polygon(corners_b[j, :2, 4:].T)

            # check is valid,  A valid Polygon may not possess any overlapping
            # exterior or interior rings.
            if bottom_a.is_valid and bottom_b.is_valid:
                bottom_overlap = bottom_a.intersection(bottom_b).area
            else:
                bottom_overlap = 0.0

            overlap3d = bottom_overlap * z_overlap
            union3d = (
                bottom_a.area * (max_z_a[i] - min_z_a[i])
                + bottom_b.area * (max_z_b[j] - min_z_b[j])
                - overlap3d
            )
            iou_3d[i, j] = overlap3d / union3d
            iou_bev[i, j] = bottom_overlap / (
                bottom_a.area + bottom_b.area - bottom_overlap
            )

    if need_bev:
        return iou_3d, iou_bev

    return iou_3d


def get_iou3d_single(box_a, box_b, need_bev=False):
    """Compute IoU between two boxes.

    Args:
        box_a (array[7]): (x, y, z, l, w, h, theta)
        box_b (array[7]): (x, y, z, l, w, h, theta)
        need_bev (bool): True to also return IoU of BEV

    Returns:
        iou_3d (float)
    """
    return get_iou3d(box_a[np.newaxis], box_b[np.newaxis], need_bev)


def _boxes_jrdb_to_pointrcnn_convention(boxes):
    # NOTE don't really know why, but seems like iou3d expects KITTI box with
    # swapped x and z axis, otherwise cannot pass unittest test_iou3d() and test_nms3d()
    boxes_kitti = convert_boxes_jrdb_to_kitti_torch(boxes)
    tmp_r2 = boxes_kitti[:, 2].clone()
    boxes_kitti[:, 2] = boxes_kitti[:, 0]
    boxes_kitti[:, 0] = tmp_r2

    return boxes_kitti


def get_iou3d_gpu(boxes, query_boxes):
    """Compute pair-wise IoU between two group of boxes.

    Args:
        boxes (tensor[N, 7]): (x, y, z, l, w, h, theta)
        query_boxes (tensor[M, 7])

    Returns:
        iou_3d (tensor[N, M])
    """
    boxes = _boxes_jrdb_to_pointrcnn_convention(boxes)
    query_boxes = _boxes_jrdb_to_pointrcnn_convention(query_boxes)
    iou_3d = iou3d.boxes_iou3d_gpu(boxes, query_boxes)

    return iou_3d


def get_iou_bev_gpu(boxes, query_boxes):
    boxes = _boxes_jrdb_to_pointrcnn_convention(boxes)
    query_boxes = _boxes_jrdb_to_pointrcnn_convention(query_boxes)
    iou_bev = iou3d.boxes_iou_bev(
        iou3d.boxes3d_to_bev_torch(boxes), iou3d.boxes3d_to_bev_torch(query_boxes)
    )

    return iou_bev


def nms_3d(boxes, scores, iou_thresh=0.3):
    """IoU based NMS

    Args:
        boxes (array[B, 7]): (x, y, z, l, w, h, theta)
        scores (array[B]): cls score for each box
        iou_thresh (float):

    Returns:
        inds (array[N]): The inds of non-duplicate box, N <= B (see np.unique)
        inverse_map (array[B]): For each box, return the index of its parents
    """
    ious = get_iou3d(boxes, boxes)  # (B, B)
    sort_inds = np.argsort(scores)[::-1]

    inds = []
    inverse_map = -np.ones(ious.shape[0], dtype=np.int32)

    for i in sort_inds:
        if inverse_map[i] == -1:
            dup_mask = np.logical_and(ious[i] > iou_thresh, inverse_map == -1)
            inverse_map[dup_mask] = i
            inds.append(i)

    return np.array(inds, dtype=np.int32), inverse_map


def nms_3d_gpu(boxes, scores, iou_thresh=0.3):
    """IoU based NMS

    Args:
        boxes (tensor[B, 7]): (x, y, z, l, w, h, theta)
        scores (tensor[B]): cls score for each box
        iou_thresh (float):

    Returns:
        inds (tensor[N]): The inds of non-duplicate box, N <= B (see np.unique)
    """
    boxes_bev = iou3d.boxes3d_to_bev_torch(_boxes_jrdb_to_pointrcnn_convention(boxes))
    return iou3d.nms_gpu(boxes_bev, scores, iou_thresh)


def nms_3d_dist(boxes, scores, l_ave=0.9, w_ave=0.5, nms_thresh=0.4):
    """Distance based NMS

    Args:
        boxes (array[B, 7]): (x, y, z, l, w, h, theta)
        scores (array[B]): cls score for each box
        l_ave (float): average box length
        w_ave (float): average box width
        nms_thresh (float):

    Returns:
        inds (array[N]): The inds of non-duplicate box, N <= B (see np.unique)
        inverse_map (array[B]): For each box, return the index of its parents
    """
    x_diff = boxes[:, 0, np.newaxis] - boxes[np.newaxis, :, 0]
    y_diff = boxes[:, 1, np.newaxis] - boxes[np.newaxis, :, 1]
    d_diff = np.hypot(x_diff, y_diff)
    t_diff = np.arctan2(y_diff, x_diff)

    cos_t = np.abs(np.cos(boxes[:, 6, np.newaxis] - t_diff))  # this is not symmetric
    d_norm = w_ave + 0.5 * (cos_t + cos_t.T) * (l_ave - w_ave)

    d_ratio = d_diff / d_norm
    sort_inds = np.argsort(scores)[::-1]

    inds = []
    inverse_map = -np.ones(d_ratio.shape[0], dtype=np.int32)

    for i in sort_inds:
        if inverse_map[i] == -1:
            dup_mask = np.logical_and(d_ratio[i] < nms_thresh, inverse_map == -1)
            inverse_map[dup_mask] = i
            inds.append(i)

    return np.array(inds, dtype=np.int32), inverse_map


def nms_3d_dist_gpu(boxes, scores, l_ave=0.9, w_ave=0.5, nms_thresh=0.4):
    """IoU based NMS

    Args:
        boxes (tensor[B, 7]): (x, y, z, l, w, h, theta)
        scores (tensor[B]): cls score for each box
        l_ave (float): average box length
        w_ave (float): average box width
        nms_thresh (float):

    Returns:
        inds (tensor[N]): The inds of non-duplicate box, N <= B (see np.unique)
    """
    boxes_bev = iou3d.boxes3d_to_bev_torch(_boxes_jrdb_to_pointrcnn_convention(boxes))
    return iou3d.nms_dist_gpu(boxes_bev, scores, l_ave, w_ave, nms_thresh)


def _ravel_hash_vec(arr):
    assert arr.ndim == 2
    arr = arr - arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def get_unique_rows(arr):
    keys = _ravel_hash_vec(arr)
    _, inds = np.unique(keys, return_index=True)
    return inds


class Box3d:
    def __init__(self, xyz, lwh, rot_z, box_id):
        """A 3D bounding box.

        Args:
            xyz (np.array(3,)): center xyz
            lwh (np.array(3,)): length, width, height. When rot_z is zero, length
                should be aligned with x axis
            rot_z (float): rotation around z axis
            box_id (int): label id used for tracking
        """
        self._xyz = xyz.reshape(-1)
        self._lwh = lwh.reshape(-1)
        self._rot_z = rot_z
        self._id = box_id

    def to_array(self):
        return np.concatenate([self._xyz, self._lwh, [self._rot_z]], axis=0)

    def get_xyz(self):
        return self._xyz

    def get_lwh(self):
        return self._lwh

    def get_id(self):
        return self._id

    def get_R(self):
        """See get_R()"""
        box = self.to_array()
        R = get_R(box[np.newaxis, :])

        return R[0]

    def to_corners(self, resize_factor=1.0, connect_inds=False):
        """See boxes_to_corners()"""
        box = self.to_array()

        if connect_inds:
            corners_xyz, connect_inds = boxes_to_corners(
                box[np.newaxis, :], resize_factor=resize_factor, connect_inds=True
            )
            return corners_xyz[0], connect_inds
        else:
            corners_xyz = boxes_to_corners(
                box[np.newaxis, :], resize_factor=resize_factor, connect_inds=False
            )
            return corners_xyz[0]

    def draw_bev(self, ax, c="red"):
        corner_xyz = self.to_corners()

        # side and back boarder
        xy = corner_xyz[:2, [1, 2, 3, 0]]
        ax.plot(xy[0], xy[1], c=c, linestyle="-")

        # front boarder
        xy = corner_xyz[:2, [0, 1]]
        ax.plot(xy[0], xy[1], c=c, linestyle="--")

    def draw_fpv(self, ax, dim, c="red"):
        """Plot first person view.

        Args:
            ax: Axes handle of matplotlib
            dim (int): 0 (x) for xz plot or 1 (y) for yz plot
            c (str, optional): color. Defaults to "red".
        """
        corner_xyz = self.to_corners()

        # top
        x = corner_xyz[dim, [0, 1, 2, 3, 0]]
        z = corner_xyz[2, [0, 1, 2, 3, 0]]
        ax.plot(x, z, c=c, linestyle="-")

        # bottom
        x = corner_xyz[dim, [4, 5, 6, 7, 4]]
        z = corner_xyz[2, [4, 5, 6, 7, 4]]
        ax.plot(x, z, c=c, linestyle="-")

        # vertical bar
        for i in range(4):
            x = corner_xyz[dim, [i, i + 4]]
            z = corner_xyz[2, [i, i + 4]]
            ax.plot(x, z, c=c, linestyle="-")

        # mark orientation
        x = corner_xyz[dim, [0, 5]]
        z = corner_xyz[2, [0, 5]]
        ax.plot(x, z, c=c, linestyle="--")
        x = corner_xyz[dim, [1, 4]]
        z = corner_xyz[2, [1, 4]]
        ax.plot(x, z, c=c, linestyle="--")


"""
Conversion API between JRDB and KITTI
"""


def convert_boxes_jrdb_to_kitti(jrdb_boxes):
    """Convert JRDB boxes to KITTI boxes

    x' = -y
    y' = -z + h / 2
    z' = x
    h' = h
    w' = w
    l' = l
    theta' = -theta

    Args:
        jrdb_boxes (array[B, 7]): x, y, z, l, w, h, theta

    Returns:
        kitti_boxes (array[B, 7]): x', y', z', h', w', l', theta'
    """
    # NOTE: Use test/test_iou3d() to check that this conversion is done correctly.
    kitti_boxes = np.stack(
        [
            -jrdb_boxes[:, 1],
            -jrdb_boxes[:, 2] + 0.5 * jrdb_boxes[:, 5],
            jrdb_boxes[:, 0],
            jrdb_boxes[:, 5],
            jrdb_boxes[:, 4],
            jrdb_boxes[:, 3],
            -jrdb_boxes[:, 6],
        ],
        axis=1,
    )

    return kitti_boxes


def convert_boxes_jrdb_to_kitti_torch(jrdb_boxes):
    kitti_boxes = torch.stack(
        [
            -jrdb_boxes[:, 1],
            -jrdb_boxes[:, 2] + 0.5 * jrdb_boxes[:, 5],
            jrdb_boxes[:, 0],
            jrdb_boxes[:, 5],
            jrdb_boxes[:, 4],
            jrdb_boxes[:, 3],
            -jrdb_boxes[:, 6],
        ],
        dim=1,
    )

    return kitti_boxes


def convert_boxes_kitti_to_jrdb(kitti_boxes):
    """Convert KITTI boxes to JRDB boxes

    x = z'
    y = -x'
    z = -y' + h' / 2
    l = l'
    w = w'
    h = h'
    theta = -theta'

    Args:
        kitti_boxes (array[B, 7]): x', y', z', h', w', l', theta'

    Returns:
        jrdb_boxes (array[B, 7]): x, y, z, l, w, h, theta
    """
    jrdb_boxes = np.stack(
        [
            kitti_boxes[:, 2],
            -kitti_boxes[:, 0],
            -kitti_boxes[:, 1] + 0.5 * kitti_boxes[:, 3],
            kitti_boxes[:, 5],
            kitti_boxes[:, 4],
            kitti_boxes[:, 3],
            -kitti_boxes[:, 6],
        ],
        axis=1,
    )

    return jrdb_boxes


def convert_boxes_kitti_to_jrdb_torch(kitti_boxes):
    jrdb_boxes = torch.stack(
        [
            kitti_boxes[:, 2],
            -kitti_boxes[:, 0],
            -kitti_boxes[:, 1] + 0.5 * kitti_boxes[:, 3],
            kitti_boxes[:, 5],
            kitti_boxes[:, 4],
            kitti_boxes[:, 3],
            -kitti_boxes[:, 6],
        ],
        dim=1,
    )

    return jrdb_boxes


"""
Dataset IO API
"""


def box_from_jrdb(jrdb_label, fast_mode=True):
    xyz_lwh_rot = np.array(
        [
            jrdb_label["box"]["cx"],
            jrdb_label["box"]["cy"],
            jrdb_label["box"]["cz"],
            jrdb_label["box"]["l"],
            jrdb_label["box"]["w"],
            jrdb_label["box"]["h"],
            jrdb_label["box"]["rot_z"],
        ],
        dtype=np.float32,
    )
    box_id = int(jrdb_label["label_id"].split(":")[-1])

    if fast_mode:
        return xyz_lwh_rot, box_id
    else:
        return Box3d(xyz_lwh_rot[:3], xyz_lwh_rot[3:6], xyz_lwh_rot[6], box_id)


def box_from_nuscenes(ann, fast_mode=True):
    xyz = ann["translation"]
    wlh = ann["size"]
    qt = Quaternion(ann["rotation"])
    # Rotation is only along vertical axis, but the rotation axis direction is
    # arbitary (plus or minus)
    r = -qt.axis[2] * qt.radians  # the minus sign is confirmed by plotting
    if abs(qt.axis[2]) != 1.0 and np.abs(qt.axis).sum() != 0.0:
        print(ann)
    box_id = ann["instance_token"]

    xyz_lwh_rot = np.array([*xyz, wlh[1], wlh[0], wlh[2], r], dtype=np.float32)

    if fast_mode:
        return xyz_lwh_rot, box_id
    else:
        return Box3d(xyz_lwh_rot[:3], xyz_lwh_rot[3:6], xyz_lwh_rot[6], box_id)


def box_to_nuscenes(box, score, sample_token, detection_name):
    xyz = box[:3].tolist()
    wlh = box[[4, 3, 5]].tolist()
    # the minus sign should be consistent with box_from_nuscenes()
    qt = Quaternion(axis=[0, 0, 1], angle=-box[6], dtype=float)

    sr_dict = {
        "sample_token": sample_token,
        "translation": xyz,
        "size": wlh,
        "rotation": list(qt),
        "velocity": [0.0, 0.0],
        "detection_name": detection_name,
        "detection_score": float(score),
        "attribute_name": "",
    }

    return sr_dict


def boxes_to_string(boxes, scores, jrdb_format=True):
    """Obtain a KITTI format string for storing detection result

    Args:
        boxes (array[B, 7])
        scores (array[B])
        jrdb_format (bool): True if the input boxes is in JRDB convention, False
            if in KITTI convention

    Returns:
        s (str)
    """
    if jrdb_format:
        if isinstance(boxes, np.ndarray):
            boxes = convert_boxes_jrdb_to_kitti(boxes)
        else:
            boxes = convert_boxes_jrdb_to_kitti_torch(boxes)

    s = ""
    for box, score in zip(boxes, scores):
        s += (
            f"Pedestrian 0 0 -1 0 -1 -1 -1 -1 {box[3]} {box[4]} {box[5]} "
            f"{box[0]} {box[1]} {box[2]} {box[6]} {score}\n"
        )
    s = s.strip("\n")

    return s


def string_to_boxes(s, jrdb_format=True, get_num_points=False):
    """Convert a KITTI format string to boxes and detection scores

    Args:
        s (str)
        jrdb_format (bool): True if the output boxes should be in JRDB convention,
            False if in KITTI convention
        get_num_points (bool): True to also return number of points. Useful
            for loading annotations

    Returns:
        boxes (array[B, 7])
        scores (array[B])
    """
    boxes = []
    scores = []

    lines = s.split("\n")
    for line in lines:
        if len(line) == 0:
            continue
        v_list = [float(v) for v in line.split()[-8:]]
        scores.append(v_list[-1])
        boxes.append([v_list[i] for i in [3, 4, 5, 0, 1, 2, 6]])

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    if jrdb_format and len(boxes) > 0:
        boxes = convert_boxes_kitti_to_jrdb(boxes)

    if get_num_points:
        num_points = np.array([int(line.split()[3]) for line in lines if len(line) > 0])
        return boxes, scores, num_points
    else:
        return boxes, scores
