import numpy as np
import torch
from torchsparse import SparseTensor
import lidar_det.utils.utils_box3d as ub3d


__all__ = [
    "collate_sparse_tensors",
    "encode_angle",
    "decode_angle",
    "decode_angle_torch",
    # "encode_canonical",
    # "encode_boxes_to_target",
    # "decode_target_to_boxes",
    # "decode_target_to_boxes_torch",
    # "get_prediction_target",
    "boxes_to_target",
    "target_to_boxes",
    "target_to_boxes_torch"
]


def collate_sparse_tensors(batch):
    """Collate sparse tensors, tensor features and coordinates must be numpy array

    Reference:
    https://github.com/mit-han-lab/torchsparse/blob/d022d4b070d0747c7ce8e8c0d93240eb5ba64995/torchsparse/utils/helpers.py#L192

    Args:
        batch (list[SparseTensors])

    Returns:
        SparseTensors
    """
    batch_inds = []
    F_list = []
    C_list = []
    for i in range(len(batch)):
        n = batch[i].F.shape[0]
        F_list.append(batch[i].F)
        C_list.append(batch[i].C)
        batch_inds.append(i * np.ones((n, 1), dtype=np.int32))

    F_batch = np.concatenate(F_list, axis=0)
    C_batch = np.concatenate(C_list, axis=0)
    batch_inds = np.concatenate(batch_inds, axis=0)
    C_batch = np.concatenate((C_batch, batch_inds), axis=1)

    return SparseTensor(
        torch.from_numpy(F_batch).float(), torch.from_numpy(C_batch).int(), batch[0].s
    )


def encode_angle(angs, num_theta_bins):
    """Encode angles as classification and regression

    Args:
        angs (array[...]): Arbitary shape, each element represents an angle
        num_theta_bins (int): Number of angular bins

    Return:
        target_angs (array[..., C]): cls + reg (1 + num_theta_bins) target encoding
    """
    # encode angle
    bin_size = 2 * np.pi / num_theta_bins
    bin_inds = ((angs % (2.0 * np.pi)) / bin_size).astype(np.int32)

    bin_center = bin_size * (np.arange(num_theta_bins) + 0.5)
    bin_res = angs[..., np.newaxis] - bin_center
    # bin_res = angs - (bin_inds.astype(np.float32) + 0.5) * bin_size
    # bin_res = bin_res[..., np.newaxis]

    # normalize angle (JRDB angle is not normalized)
    bin_res = bin_res % (2.0 * np.pi)
    bin_res[bin_res > np.pi] -= 2.0 * np.pi

    target_angs = np.concatenate((bin_inds[..., np.newaxis], bin_res), axis=-1)

    return target_angs


def decode_angle(target_angs, num_theta_bins):
    """Decode angles from classification and regression results

    Args:
        target_angs (array[..., C]): cls + reg (1 + num_theta_bins) target encoding
        num_theta_bins (int): Number of angular bins

    Returns:
        angs (array[...]): Same shape as `bin_inds`
    """
    if target_angs.shape[-1] == 2 * num_theta_bins:
        # if target_angs.shape[-1] == num_theta_bins + 1:
        bin_inds = np.argmax(target_angs[..., :num_theta_bins], axis=-1)
        bin_res = target_angs[..., num_theta_bins:]
    else:
        bin_inds = target_angs[..., 0]
        bin_res = target_angs[..., 1:]

    # orientation
    bin_size = 2.0 * np.pi / num_theta_bins
    bin_res_i = bin_res.reshape(-1, num_theta_bins)[
        np.arange(bin_inds.size), bin_inds.reshape(-1).astype(np.int64)
    ].reshape(bin_inds.shape)
    # bin_res_i = bin_res[..., 0]

    angs = (bin_inds + 0.5) * bin_size + bin_res_i  # (-pi, 3*pi)
    angs[angs > np.pi] -= 2.0 * np.pi

    return angs


def decode_angle_torch(target_angs, num_theta_bins):
    """Decode angles from classification and regression results

    Args:
        target_angs (tensor[..., C]): cls + reg (1 + num_theta_bins) target encoding
        bin_cls_score (bool): True if the bin index is given as probability for
            each bin

    Returns:
        angs (tensor[...]): Same shape as `bin_inds`
    """
    if target_angs.shape[-1] == 2 * num_theta_bins:
        # if target_angs.shape[-1] == num_theta_bins + 1:
        bin_inds = target_angs[..., :num_theta_bins].argmax(dim=-1)
        bin_res = target_angs[..., num_theta_bins:]
    else:
        bin_inds = target_angs[..., 0]
        bin_res = target_angs[..., 1:]

    # orientation
    bin_size = 2.0 * np.pi / num_theta_bins
    bin_res_i = bin_res.view(-1, num_theta_bins)[
        torch.arange(bin_inds.nelement()), bin_inds.view(-1).long()
    ].reshape(bin_inds.shape)
    # bin_res_i = bin_res[..., 0]

    angs = (bin_inds + 0.5) * bin_size + bin_res_i
    angs[angs > np.pi] -= 2.0 * np.pi

    return angs


def get_R(angles):
    """Get rotation matrix R along verticle axis from the angle

    Args:
        angles (array[N]): In radian

    Returns:
        Rs (array[N, 3, 3])
    """
    cs, ss = np.cos(angles), np.sin(angles)
    zeros, ones = np.zeros(len(cs)), np.ones(len(cs))
    Rs = np.array(
        [[cs, ss, zeros], [-ss, cs, zeros], [zeros, zeros, ones]], dtype=np.float32
    )  # (3, 3, N)

    return Rs.transpose((2, 0, 1))


def get_R_torch(angles):
    """Get rotation matrix R along verticle axis from the angle

    Args:
        angles (tensor[N]): In radian

    Returns:
        Rs (tensor[N, 3, 3])
    """
    cs, ss = torch.cos(angles), torch.sin(angles)
    zeros = torch.zeros(len(cs), device=angles.device)
    ones = torch.ones(len(cs), device=angles.device)
    Rs = torch.empty((angles.shape[0], 3, 3), device=angles.device).float()  # (N, 3, 3)
    Rs[:, 0] = torch.stack((cs, ss, zeros), dim=1)
    Rs[:, 1] = torch.stack((-ss, cs, zeros), dim=1)
    Rs[:, 2] = torch.stack((zeros, zeros, ones), dim=1)

    return Rs


def encode_canonical(pc, vec, ang):
    """Convert a vector and an angle to canonical frame defined by the scan point

    Args:
        pc (array[3, N]): point cloud xyz
        vec (array[3, N]): vector xyz to be transformed
        ang (array[N, M]): rotation angles to be transformed, around vertical axis,
            M angles for each point

    Returns:
        vec_cano (array[3, N])
        ang_cano (array[N, M])
    """
    if len(ang.shape) == 1:
        ang = ang[:, np.newaxis]

    theta = np.arctan2(pc[1], pc[0])
    R = get_R(theta)  # p_canonical = R * p_world
    vec_cano = R @ vec.T[..., np.newaxis]  # (N, 3, 1)
    vec_cano = vec_cano[:, :, 0].T
    ang_cano = ang - theta[:, np.newaxis]

    return vec_cano, ang_cano


def decode_canonical(pc, vec, ang):
    """Decode vectors and angles from canonical frame defined by the scan point

    Args:
        pc (array[3, N]): point cloud xyz
        vec (array[N, A, 3]): vector xyz to be transformed
        ang (array[N, A]): rotation angles to be transformed, around vertical axis,
            M angles for each point

    Returns:
        vec_w (array[N, A, 3])
        ang_w (array[N, A])
    """
    theta = np.arctan2(pc[1], pc[0])
    R = get_R(theta)  # p_canonical = R * p_world, (N, 3, 3)
    R = np.repeat(R[:, np.newaxis, :, :], vec.shape[1], axis=1)  # (N, A, 3, 3)
    vec_w = R.transpose((0, 1, 3, 2)) @ vec[..., np.newaxis]  # (N, A, 3, 1)
    vec_w = vec_w[:, :, :, 0]
    ang_w = ang + theta[:, np.newaxis]

    return vec_w, ang_w


def decode_canonical_torch(pc, vec, ang):
    """Decode vectors and angles from canonical frame defined by the scan point

    Args:
        pc (tensor[3, N]): point cloud xyz
        vec (tensor[N, A, 3]): vector xyz to be transformed
        ang (tensor[N, A]): rotation angles to be transformed, around vertical axis,
            M angles for each point

    Returns:
        vec_w (tensor[N, A, 3])
        ang_w (tensor[N, A])
    """
    theta = torch.atan2(pc[1], pc[0])
    R = get_R_torch(theta)  # p_canonical = R * p_world, (N, 3, 3)
    R = R.unsqueeze(dim=1).repeat(1, vec.shape[1], 1, 1)  # (N, A, 3, 3)
    vec_w = R.transpose(3, 2) @ vec.unsqueeze(dim=-1)  # (N, A, 3, 1)
    vec_w = vec_w[:, :, :, 0]
    ang_w = ang + theta.unsqueeze(dim=-1)

    return vec_w, ang_w


def encode_boxes_to_target(
    pc, boxes, ave_lwh, num_anchors=2, num_theta_bins=12, canonical=False
):
    """Generate per point regression target from points and matching boxes

    Args:
        pc (array[3, N]): Points
        boxes (array[7, N]): Target boxes
        ave_lwh (tuple[3]): average lwh value, used as anchor to encode actual value
        num_anchors (int): Number of anchor boxes
        num_theta_bins (int): How many bins to encode orientation

    Returns:
        target (array[N, NA, C])
    """
    N = boxes.shape[1]
    NA = num_anchors

    # location and dimension
    target_xyz = boxes[:3] - pc  # (3, N)
    target_dim = boxes[3:6] / np.array(ave_lwh, dtype=np.float32).reshape(3, 1)
    target_dim = np.log(target_dim)

    anchor_theta = np.arange(NA, dtype=np.float32) * (np.pi / NA)
    res_theta = boxes[6].reshape(N, -1) - anchor_theta.reshape(-1, NA)  # (N, NA)

    if canonical:
        target_xyz, res_theta = encode_canonical(pc, target_xyz, res_theta)

    # encode angle
    if num_theta_bins > 1:
        target_theta = encode_angle(res_theta, num_theta_bins)  # (N, NA, C)
    else:
        target_theta = res_theta[..., np.newaxis]  # (N, NA, 1)

    target_xyz = np.tile(target_xyz.T[:, np.newaxis, :], (1, NA, 1))  # (N, NA, 3)
    target_dim = np.tile(target_dim.T[:, np.newaxis, :], (1, NA, 1))

    target = np.concatenate(
        (target_xyz, target_dim, target_theta), axis=2
    )  # (N, NA, 3 + 3 + 1 + num_theta_bins)

    return target


def decode_target_to_boxes(pc, target, ave_lwh, num_theta_bins, canonical=False):
    """Decode per point regression target to box parameters

    Args:
        pc (array[3, N]): Points
        target (array[N, NA, C]): Regression target
        ave_lwh (tuple[3]): average lwh value, used as anchor to encode actual value
        num_theta_bins (int): How many bins to encode orientation

    Returns:
        boxes (array[N, NA, 7]): Target boxes
    """
    if num_theta_bins > 1:
        boxes_ori = decode_angle(target[:, :, 6:], num_theta_bins)
    else:
        boxes_ori = target[:, :, 6]
    NA = target.shape[1]
    anchor_theta = np.arange(NA, dtype=np.float32) * (np.pi / NA)
    boxes_ori = boxes_ori + anchor_theta.reshape(1, NA)  # (N, NA)

    if canonical:
        target_xyz, boxes_ori = decode_canonical(pc, target[:, :, :3], boxes_ori)
    else:
        target_xyz = target[:, :, :3]

    # location and dimension
    boxes_xyz = target_xyz + pc.T[:, np.newaxis, :]  # (N, NA, 3)
    boxes_dim = np.exp(target[:, :, 3:6]) * np.array(ave_lwh, dtype=np.float32).reshape(
        1, 1, 3
    )

    boxes = np.concatenate(
        (boxes_xyz, boxes_dim, boxes_ori[..., np.newaxis]), axis=2
    )  # (N, NA, 7)

    return boxes


def decode_target_to_boxes_torch(pc, target, ave_lwh, num_theta_bins, canonical=False):
    """Decode per point regression target to box parameters

    Args:
        pc (tensor[3, N]): Points
        target (tensor[N, NA, C]): Regression target
        ave_lwh (tuple[3]): average lwh value, used as anchor to encode actual value
        num_theta_bins (int): How many bins to encode orientation

    Returns:
        boxes (tensor[N, NA, 7]): Target boxes
    """
    if num_theta_bins > 1:
        boxes_ori = decode_angle_torch(target[:, :, 6:], num_theta_bins)
    else:
        boxes_ori = target[:, :, 6]
    NA = target.shape[1]
    anchor_theta = torch.arange(NA, device=pc.device).float() * (np.pi / NA)
    boxes_ori += anchor_theta.view(1, NA)  # (N, NA)

    if canonical:
        target_xyz, boxes_ori = decode_canonical_torch(pc, target[:, :, :3], boxes_ori)
    else:
        target_xyz = target[:, :, :3]

    # location and dimension
    boxes_xyz = target_xyz + pc.T.unsqueeze(dim=1)  # (N, NA, 3)
    ave_lwh = torch.tensor(ave_lwh, device=pc.device).float().view(1, 1, 3)
    boxes_dim = torch.exp(target[:, :, 3:6]) * ave_lwh

    boxes = torch.cat(
        (boxes_xyz, boxes_dim, boxes_ori.unsqueeze(dim=-1)), dim=2
    )  # (N, NA, 7)

    return boxes


def boxes_to_target(boxes, ave_lwh, num_anchors=2, num_theta_bins=12):
    """Generate per point regression target from points and matching boxes

    Args:
        boxes (array[B, 7]): x, y, z, l, w, h, theta
        ave_lwh (tuple[3]): average lwh value, used as anchor to encode actual value
        num_anchors (int): Number of anchor boxes
        num_theta_bins (int): How many bins to encode orientation

    Returns:
        target (array[B, NA, C])
    """
    B = boxes.shape[0]
    A = num_anchors

    # location and dimension
    target_xyz = boxes[:, :3]
    target_dim = boxes[:, 3:6] / np.array(ave_lwh, dtype=np.float32).reshape(1, 3)
    target_dim = np.log(target_dim)

    anchor_theta = np.arange(A, dtype=np.float32) * (np.pi / A)
    res_theta = boxes[:, 6].reshape(B, 1) - anchor_theta.reshape(1, A)  # (B, A)

    # encode angle
    if num_theta_bins > 1:
        target_theta = encode_angle(res_theta, num_theta_bins)  # (B, A, C)
    else:
        target_theta = res_theta[..., np.newaxis]  # (B, A, 1)

    target_xyz = np.tile(target_xyz[:, np.newaxis, :], (1, A, 1))  # (B, A, 3)
    target_dim = np.tile(target_dim[:, np.newaxis, :], (1, A, 1))

    target = np.concatenate(
        (target_xyz, target_dim, target_theta), axis=2
    )  # (B, A, 3 + 3 + 1 + num_theta_bins)

    return target


def target_to_boxes(target, ave_lwh, num_theta_bins):
    """Decode per point regression target to box parameters

    Args:
        target (array[B, A, C]): Regression target
        ave_lwh (tuple[3]): average lwh value, used as anchor to encode actual value
        num_theta_bins (int): How many bins to encode orientation

    Returns:
        boxes (array[B, A, 7]): Target boxes
    """
    if num_theta_bins > 1:
        boxes_ori = decode_angle(target[:, :, 6:], num_theta_bins)  # (B, A)
    else:
        boxes_ori = target[:, :, 6]
    A = target.shape[1]
    anchor_theta = np.arange(A, dtype=np.float32) * (np.pi / A)
    boxes_ori = boxes_ori + anchor_theta.reshape(1, A)  # (B, A)

    # location and dimension
    boxes_xyz = target[:, :, :3]  # (B, A, 3)
    boxes_dim = np.exp(target[:, :, 3:6]) * np.array(ave_lwh, dtype=np.float32).reshape(
        1, 1, 3
    )

    boxes = np.concatenate(
        (boxes_xyz, boxes_dim, boxes_ori[..., np.newaxis]), axis=2
    )  # (B, A, 7)

    return boxes


def target_to_boxes_torch(target, ave_lwh, num_theta_bins):
    """Decode per point regression target to box parameters

    Args:
        target (tensor[B, A, C]): Regression target
        ave_lwh (tuple[3]): average lwh value, used as anchor to encode actual value
        num_theta_bins (int): How many bins to encode orientation

    Returns:
        boxes (tensor[B, A, 7]): Target boxes
    """
    if num_theta_bins > 1:
        boxes_ori = decode_angle_torch(target[:, :, 6:], num_theta_bins)  # (B, A)
    else:
        boxes_ori = target[:, :, 6]
    A = target.shape[1]
    anchor_theta = torch.arange(A, device=target.device).float() * (np.pi / A)
    boxes_ori += anchor_theta.view(1, A)  # (N, NA)

    # location and dimension
    boxes_xyz = target[:, :, :3]
    ave_lwh = torch.tensor(ave_lwh, device=target.device).float().view(1, 1, 3)
    boxes_dim = torch.exp(target[:, :, 3:6]) * ave_lwh

    boxes = torch.cat(
        (boxes_xyz, boxes_dim, boxes_ori.unsqueeze(dim=-1)), dim=2
    )  # (B, A, 7)

    return boxes


def get_cls_target(
    pc, boxes, ave_lwh, num_anchors=2, target_mode=0, dist_min=0.7, dist_max=1.5,
):
    """Return per point regression label

    Args:
        pc (array[3, N]): xyz point cloud
        boxes (array[B, 7]): bounding boxes
        ave_lwh (tuple[3]): average lwh value, used as anchor to encode actual value
        num_anchors (int): Number of anchor boxes
        num_theta_bins (int): How many bins to encode orientation
        target_mode (int): How cls targets are encoded,
            0-inclusion, 1-anchors, 2-distance

    Returns:
        cls_target (array[N, NA])
        closest_box_inds (array[N,]): The index of the target box for each point
    """
    N = pc.shape[1]
    assert boxes.shape[0] > 0

    # cls label
    if target_mode >= 2:
        closest_box_inds, closest_box_dist = ub3d.find_closest_boxes(pc, boxes)
        cls_target = 1.0 - (closest_box_dist - dist_min) / (dist_max - dist_min)
        cls_target = np.clip(cls_target, a_min=0.0, a_max=1.0)  # (N)
        cls_target = np.repeat(cls_target[:, np.newaxis], num_anchors, axis=1)
    elif target_mode == 1:
        # anchor boxes
        anchors = np.empty((N, num_anchors, 7), dtype=np.float32)
        anchors[:, :, :3] = pc.T[:, np.newaxis, :3]
        anchors[:, :, 3:6] = np.array(ave_lwh).reshape(1, 1, 3)
        anchor_theta = np.arange(num_anchors, dtype=np.float32) * (np.pi / num_anchors)
        anchors[:, :, 6] = anchor_theta

        # assign cls based on iou with gt boxes
        closest_box_inds, closest_box_dist = ub3d.find_closest_boxes(pc, boxes)
        cls_target = np.zeros((N, num_anchors), dtype=np.float32)
        pts_inds = np.arange(N)[
            closest_box_dist < 0.75
        ]  # don't compute iou for far points to save computation
        for p_i in pts_inds:
            box_gt = boxes[closest_box_inds[p_i]]
            ious = ub3d.get_iou3d(anchors[p_i], box_gt[np.newaxis]).reshape(-1)  # (A)
            cls_target[p_i, ious >= 0.35] = -1
            cls_target[p_i, ious >= 0.5] = 1

    #    # assign cls based on iou with gt boxes
    #     closest_box_inds, _ = ub3d.find_closest_boxes(pc, boxes)
    #     cls_target = -1 * np.zeros((N, num_anchors), dtype=np.float32)
    #     ious = ub3d.get_iou3d_gpu(
    #         torch.from_numpy(anchors.reshape(N * num_anchors, 7))
    #         .cuda(non_blocking=True)
    #         .float(),
    #         torch.from_numpy(boxes).cuda(non_blocking=True).float(),
    #     ).view(N, num_anchors, boxes.shape[0])
    #     ious = ious.max(dim=2).data.cpu().numpy()  # (N, A)
    #     cls_target[ious >= 0.5] = 1
    #     cls_target[ious <= 0.35] = 0
    else:
        cls_target = np.zeros((N, num_anchors), dtype=np.float32)
        fg_mask, closest_box_inds = ub3d.find_in_box_points(pc, boxes)
        cls_target[fg_mask] = 1.0  # fg target

        # exclude points around box boundary from cls training
        fg_mask_big, _ = ub3d.find_in_box_points(pc, boxes, resize_factor=1.1)
        fg_mask_big = np.logical_and(fg_mask_big, np.logical_not(fg_mask))
        cls_target[fg_mask_big] = -1.0  # ignore

    return cls_target, closest_box_inds


def get_prediction_target(
    pc,
    boxes,
    ave_lwh,
    boxes_cls=None,
    num_anchors=2,
    num_theta_bins=12,
    canonical=False,
    target_mode=0,
    dist_min=0.7,
    dist_max=1.5,
):
    """Return per point regression label

    Args:
        pc (array[3, N]): xyz point cloud
        boxes (array[B, 7]): bounding boxes
        ave_lwh (tuple[3]): average lwh value, used as anchor to encode actual value
        num_anchors (int): Number of anchor boxes
        num_theta_bins (int): How many bins to encode orientation
        target_mode (int): How cls targets are encoded,
            0-inclusion, 1-anchors, 2-distance

    Returns:
        pred_target (array[N, NA, 1 + 7 + num_theta_bins]): Prediction target
            for each point and anchor, cls + xyz + lwh + theta_bin_inds + theta_bin_res
        closest_box_inds (array[N,]): The index of the target box for each point
        closest_boxes (array[N, 7])
    """
    N = pc.shape[1]
    multi_cls = 1 if boxes_cls is not None else 0
    if num_theta_bins > 1:
        pred_target = np.zeros(
            (N, num_anchors, 8 + num_theta_bins + multi_cls), dtype=np.float32
        )
        # pred_target = np.zeros((N, num_anchors, 9 + multi_cls), dtype=np.float32)
    else:
        pred_target = np.zeros((N, num_anchors, 8 + multi_cls), dtype=np.float32)

    if boxes.shape[0] == 0:
        return pred_target, None, None

    # objectness cls label
    pred_target[:, :, 0], closest_box_inds = get_cls_target(
        pc,
        boxes,
        ave_lwh,
        num_anchors=num_anchors,
        target_mode=target_mode,
        dist_min=dist_min,
        dist_max=dist_max,
    )

    # reg label
    closest_boxes = boxes[closest_box_inds, :]  # (N, 7)
    if multi_cls:
        pred_target[:, :, 1:-1] = encode_boxes_to_target(
            pc,
            closest_boxes.T,
            ave_lwh,
            num_anchors,
            num_theta_bins,
            canonical=canonical,
        )
        pred_target[:, :, -1] = boxes_cls[closest_box_inds, np.newaxis]
    else:
        pred_target[:, :, 1:] = encode_boxes_to_target(
            pc,
            closest_boxes.T,
            ave_lwh,
            num_anchors,
            num_theta_bins,
            canonical=canonical,
        )

    # cls label

    return pred_target, closest_box_inds, closest_boxes
