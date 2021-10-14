import os
import numpy as np
import json

# import shutil

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchsparse import SparseTensor

from jrdb_det3d_eval import eval_jrdb

# from lidar_det.dataset import decode_target_to_boxes_torch
from lidar_det.dataset import target_to_boxes_torch
import lidar_det.utils.utils_box3d as ub3d
from lidar_det.utils.viz_plt import plot_bev
from lidar_det.utils.eval_nuscenes import eval_nuscenes
from lidar_det.pipeline.loss_lib import binary_focal_loss


__all__ = ["model_fn", "model_eval_fn", "model_eval_collate_fn", "error_fn"]


# # From https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/detection  # noqa
# _NUSC_CLS_MAX_DIST = [30, 40, 50, 50, 50, 40, 40, 30, 50, 50]


def _hypot(x, y):
    # TODO use torch.hypot for PyTorch >= 1.7
    return torch.sqrt(torch.square(x) + torch.square(y))


def _pyviz_render(batch_dict):
    from lidar_det.utils.viz_pyviz import draw_lidar

    # raw points
    raw_pc = batch_dict["net_input"].F.data.cpu().numpy()
    viz = draw_lidar(pc=raw_pc, pts_name="raw_point_cloud", pts_show=True)

    # boxes
    boxes_gt = batch_dict["boxes_gt"][0]
    bmask, _ = ub3d.find_in_box_points(raw_pc.T, boxes_gt)
    viz = draw_lidar(
        pc=raw_pc[bmask],
        pts_name="boxes",
        viz=viz,
        pts_color=(1.0, 0.0, 0.0),
        pts_show=True,
    )
    viz = draw_lidar(
        pc=boxes_gt[:, :3],
        pts_name="boxes_center",
        viz=viz,
        pts_color=(0.0, 1.0, 0.0),
        pts_show=True,
        pts_size=200,
    )

    return viz


def compute_box_loss(pred, target, tb_dict, weight=None):
    # pred (N, C0), target (N, C1), fg points only

    # location and dimension loss
    loc_loss = F.mse_loss(pred[:, :3], target[:, :3], reduction="none").mean(dim=1)
    lwh_loss = F.mse_loss(pred[:, 3:6], target[:, 3:6], reduction="none").mean(dim=1)

    if weight is not None:
        loc_loss = (weight * loc_loss).mean()
        lwh_loss = (weight * lwh_loss).mean()
    else:
        loc_loss = loc_loss.mean()
        lwh_loss = lwh_loss.mean()

    tb_dict["loc_loss"] = loc_loss.item()
    tb_dict["lwh_loss"] = lwh_loss.item()
    box_loss = loc_loss + lwh_loss

    # orientation loss
    if pred.shape[1] > 7:
        NUM_BINS = int((pred.shape[1] - 6) / 2)
        # pred_bk = pred[:, 6 : 6 + NUM_BINS].data.cpu().numpy()
        # t_bk = target[:, 6].data.cpu().numpy()
        ori_cls_loss = F.cross_entropy(
            pred[:, 6 : 6 + NUM_BINS], target[:, 6].long(), reduction="none",
        )
        ori_reg_loss = F.mse_loss(
            pred[:, 6 + NUM_BINS :], target[:, 7:], reduction="none"
        ).mean(dim=1)
        # ori_reg_loss = F.mse_loss(
        #     reg_pred[fg_mask, 6 + NUM_BINS], reg_target[:, 7], reduction="none"
        # )

        if weight is not None:
            ori_cls_loss = (weight * ori_cls_loss).mean()
            ori_reg_loss = (weight * ori_reg_loss).mean()
        else:
            ori_cls_loss = ori_cls_loss.mean()
            ori_reg_loss = ori_reg_loss.mean()

        ori_cls_loss *= 0.1
        ori_reg_loss *= 0.01

        tb_dict["ori_cls_loss"] = ori_cls_loss.item()
        tb_dict["ori_reg_loss"] = ori_reg_loss.item()
        box_loss = box_loss + ori_cls_loss + ori_reg_loss

    else:
        ori_reg_loss = F.mse_loss(pred[:, 6], target[:, 6], reduction="none")

        if weight is not None:
            ori_reg_loss = (weight * ori_reg_loss).mean()
        else:
            ori_reg_loss = ori_reg_loss.mean()

        tb_dict["ori_reg_loss"] = ori_reg_loss.item()
        box_loss = box_loss + ori_reg_loss

    return box_loss, tb_dict


def compute_prediction_loss(
    voxel_center,
    cls_pred,
    reg_pred,
    inds_map,
    kf_mask,
    batch_dict,
    cls_dist_thresh,
    target_mode,
    disentangled_loss,
):
    # cls_pred (N, A, S), reg_pred (N, A, S, C), voxel_center (N, 1, 1, 3)
    tb_dict = {}

    boxes_encoded = batch_dict["boxes_encoded"]
    boxes_matched = batch_dict["boxes_matched"]
    if inds_map is not None:
        boxes_encoded = boxes_encoded[inds_map]  # (N, A, S, C)
        boxes_matched = boxes_matched[inds_map]  # (N, S, 7)
    boxes_encoded = boxes_encoded.cuda(non_blocking=True).float()
    boxes_matched = boxes_matched.cuda(non_blocking=True).float()

    # only compute lost for keyframes, used for nuScenes
    if kf_mask is not None:
        boxes_encoded = boxes_encoded[kf_mask]
        boxes_matched = boxes_matched[kf_mask]
        voxel_center = voxel_center[kf_mask]
        cls_pred = cls_pred[kf_mask]
        reg_pred = reg_pred[kf_mask]

    N, A, S = cls_pred.shape

    # # -------- DEBUG old code --------
    # # cls target
    # bv_dist = boxes_encoded[:, 0, :, :3] - voxel_center[:, 0, :, :]  # (N, S, 3)
    # bv_dist = torch.norm(bv_dist, dim=2)  # (N, S)
    # # bv_dist = _hypot(bv_dist[:, :, 0], bv_dist[:, :, 1])  # (N, S)
    # dist_min = cls_dist_thresh[:, 0].unsqueeze(dim=0)  # (1, S)
    # dist_max = cls_dist_thresh[:, 1].unsqueeze(dim=0)
    # bv_dist_normalized = (bv_dist - dist_min) / (dist_max - dist_min)
    # cls_target = 1.0 - bv_dist_normalized.clamp(0.0, 1.0)
    # cls_target = cls_target.unsqueeze(dim=1).repeat(1, A, 1)  # (N, A, S)
    # # --------

    # cls target
    pb_dist, ib_mask = ub3d.distance_pc_to_boxes_torch(
        voxel_center[:, 0, :, :],  # (N, 1, 3)
        boxes_matched,  # (N, S, 7)
        normalize=True,
        delta_size=0.2,
        return_in_box_mask=True,
    )  # (N, S)
    cls_target = ib_mask.float() * (1.0 - pb_dist.clamp(0.0, 1.0))
    cls_target = cls_target.unsqueeze(dim=1).repeat(1, A, 1)  # (N, A, S)

    # no boxes of certain cls in the scene
    empty_mask = boxes_encoded[0, 0].sum(dim=-1) == 0  # (S,)
    cls_target[:, :, empty_mask] = 0.0

    # # -------- DEBUG drawing (batch size needs to be 1) --------
    # from lidar_det.utils.viz_pyviz import draw_lidar

    # viz = _pyviz_render(batch_dict)
    # for s in range(S):
    #     ptc_s = cls_target[:, 0, s].data.cpu().numpy()
    #     pts_s = voxel_center[:, 0, 0, :].data.cpu().numpy()
    #     pmask = ptc_s > 0
    #     if pmask.max() > 0:
    #         viz = draw_lidar(
    #             pc=pts_s[pmask], pts_name=f"class_{s}", pts_color=ptc_s[pmask], viz=viz  # noqa
    #         )

    # viz.save("./logs/DEV_tests/pyviz")
    # # --------

    # cls loss
    cls_target = cls_target.view(-1)
    cls_pred = cls_pred.view(-1)
    # # -------- DEBUG calculate bce only for one-hot labels --------
    # valid_mask = torch.logical_or(cls_target == 0, cls_target == 1)
    # valid_ratio = valid_mask.sum().item() / valid_mask.__len__()
    # # assert valid_ratio > 0, "No valid points in this batch."
    # tb_dict["valid_ratio"] = valid_ratio
    # cls_loss = F.binary_cross_entropy_with_logits(
    #     cls_pred[valid_mask], cls_target[valid_mask], reduction="mean"
    # )
    # # --------
    cls_loss = F.binary_cross_entropy_with_logits(
        cls_pred, cls_target, reduction="mean"
    )
    # cls_loss = binary_focal_loss(
    #     cls_pred, cls_target, gamma=2.0, alpha=0.75, reduction="mean"
    # )
    cls_loss *= 100
    total_loss = cls_loss
    tb_dict["cls_loss"] = cls_loss.item()

    # reg loss
    fg_mask = cls_target > 0
    fg_ratio = fg_mask.sum().item() / fg_mask.__len__()
    tb_dict["fg_ratio"] = fg_ratio

    if fg_ratio <= 0.0:
        tb_dict["loss"] = total_loss.item()
        return total_loss, tb_dict

    # reg loss
    reg_weight = cls_target[fg_mask] if target_mode == 3 else None
    reg_target = boxes_encoded.view(N * A * S, -1)[fg_mask]
    reg_pred = reg_pred.view(N * A * S, -1)[fg_mask]
    box_loss, tb_dict = compute_box_loss(
        reg_pred, reg_target, tb_dict, weight=reg_weight
    )
    total_loss = total_loss + box_loss

    # # multi-class focal loss
    # if ce_pred is not None:
    #     boxes_cls = batch_dict["boxes_cls"]
    #     if inds_map is not None:
    #         boxes_cls = boxes_cls[inds_map]  # (N,)
    #     boxes_cls = boxes_cls.cuda(non_blocking=True).long()
    #     boxes_cls = boxes_cls.unsqueeze(dim=1).repeat(1, A)

    #     ce_target = boxes_cls.view(-1)[fg_mask]
    #     ce_pred = ce_pred.view(N * A, -1)[fg_mask]

    #     # focal loss
    #     ce_loss = F.cross_entropy(ce_pred, ce_target, reduction="mean")
    #     # pt = torch.exp(-ce_loss)
    #     # gamma = 2.0
    #     # ce_loss = ((1.0 - pt) ** gamma * ce_loss).mean()
    #     tb_dict["ce_loss"] = ce_loss.item()
    #     total_loss = total_loss + ce_loss

    tb_dict["loss"] = total_loss.item()

    return total_loss, tb_dict


def error_fn(model, batch_dict):
    bs = len(batch_dict["points"])
    s = f"batch size: {bs}\n"

    num_points = [pts.shape[1] for pts in batch_dict["points"]]
    s += f"number of points: {num_points}\n"

    num_voxels = batch_dict["num_voxels"]
    s += f"number of voxels: {num_voxels}\n"

    num_boxes = [boxes.shape[0] for boxes in batch_dict["boxes_gt"]]
    s += f"number of boxes: {num_boxes}"

    return s


def model_fn(model, batch_dict, target_mode, disentangled_loss, reduce_batch=False):
    # # NOTE somehow this does not reduce the memory consumption
    # if reduce_batch:
    #     kept_ratio = 0.8
    #     ori_bs = len(batch_dict["points"])
    #     new_bs = max(int(ori_bs * kept_ratio), 1)
    #     kept_mask = batch_dict["net_input"].C[:, 3] < new_bs
    #     for k, v in batch_dict.items():
    #         if isinstance(v, list):
    #             batch_dict[k] = v[:new_bs]
    #         elif isinstance(v, SparseTensor):
    #             v.F = v.F[kept_mask]
    #             v.C = v.C[kept_mask]
    #         elif isinstance(v, torch.Tensor):
    #             batch_dict[k] = v[kept_mask]

    net_input = batch_dict["net_input"].cuda()
    net_pred = model(net_input)  # (N, C)

    if isinstance(net_pred, tuple):
        net_pred, inds_map, voxel_center = net_pred
    else:
        inds_map = None
        # # -------- DEBUG use raw points for higher resolution --------
        # voxel_center = net_input.F[:, None, None, :]  # (N, 1, 1, 3)
        # # --------
        voxel_center = net_input.C[:, :3].clone().float() + 0.5

    param_dict = batch_dict["params"]
    voxel_offset = (
        torch.from_numpy(param_dict["voxel_offset"]).cuda(non_blocking=True).float()
    )
    voxel_size = (
        torch.from_numpy(param_dict["voxel_size"]).cuda(non_blocking=True).float()
    )
    voxel_center = (voxel_center - voxel_offset.view(1, 3)) * voxel_size.view(1, 3)
    voxel_center = voxel_center[:, None, None, :]  # (N, 1, 1, 3)

    net_pred = net_pred.view(
        net_pred.shape[0], model.num_anchors, model.num_classes, -1
    )  # (N, A, S, C)
    cls_pred = net_pred[..., 0]  # (N, A, S)
    reg_pred = net_pred[..., 1:]  # (N, A, S, C - 1)
    reg_pred[..., :3] = reg_pred[..., :3] + voxel_center  # reg offset from voxel center

    # keyframe mask, used for nuScenes, only supervise points in keyframe
    kf_mask = batch_dict["net_input_kfmask"]
    if isinstance(kf_mask, list):
        kf_mask = None
    else:
        if inds_map is not None:
            kf_mask = kf_mask[inds_map]
        kf_mask = kf_mask.cuda(non_blocking=True).bool()

    rtn_dict = {
        "cls_pred": cls_pred,
        "reg_pred": reg_pred,
        "inds_map": inds_map,
        "kf_mask": kf_mask,
    }

    # no label for test set, inference only
    if "boxes_encoded" not in batch_dict.keys():
        tb_dict = {}
        return 0.0, tb_dict, rtn_dict

    # compute loss
    cls_dist_thresh = param_dict["dist_thresh"]
    cls_dist_thresh = torch.tensor(cls_dist_thresh).cuda(non_blocking=True).float()
    total_loss, tb_dict = compute_prediction_loss(
        voxel_center,
        cls_pred,
        reg_pred,
        inds_map,
        kf_mask,
        batch_dict,
        cls_dist_thresh,
        target_mode,
        disentangled_loss,
    )

    return total_loss, tb_dict, rtn_dict


def model_eval_fn(
    model, batch_dict, full_eval=False, plotting=False, output_dir=None, nuscenes=False
):
    _, tb_dict, rtn_dict = model.model_fn(model, batch_dict)

    if not full_eval or output_dir is None:
        return tb_dict, {}

    cls_pred = rtn_dict["cls_pred"]  # (N, A, S)
    reg_pred = rtn_dict["reg_pred"]  # (N, A, S, C)
    inds_map = rtn_dict["inds_map"]
    kf_mask = rtn_dict["kf_mask"]
    batch_inds = batch_dict["net_input"].C[:, 3]
    if inds_map is not None:
        batch_inds = batch_inds[inds_map]  # (N, )

    # # -------- DEBUG use perfect predictions --------
    # N, A, S = cls_pred.shape

    # boxes_encoded = batch_dict["boxes_encoded"]  # (N, A, S, C)
    # if inds_map is not None:
    #     boxes_encoded = boxes_encoded[inds_map]
    # boxes_encoded = boxes_encoded.cuda(non_blocking=True).float()

    # num_theta_bins = boxes_encoded.shape[-1] - 7
    # reg_pred[..., :6] = boxes_encoded[..., :6]
    # reg_pred[..., -num_theta_bins:] = boxes_encoded[..., -num_theta_bins:]

    # cls_pred = torch.rand_like(cls_pred)
    # empty_mask = boxes_encoded[0, 0].sum(dim=-1) == 0  # (S,)
    # cls_pred[:, :, empty_mask] = 0.0
    # # --------

    # only generate boxes for keyframe points, used for nuScenes
    if kf_mask is not None:
        cls_pred = cls_pred[kf_mask]
        reg_pred = reg_pred[kf_mask]
        batch_inds = batch_inds[kf_mask]

    # postprocess network prediction to get detection
    N, A, S, C = reg_pred.shape
    cls_pred = torch.sigmoid(cls_pred)
    ave_lwh = batch_dict["params"]["ave_lwh"]
    num_theta_bins = int((C - 6) / 2)

    # go over each sample
    eval_dict = {}
    for i_batch in range(batch_inds.max() + 1):
        # go over each class
        bmask = batch_inds == i_batch  # (N, )
        boxes_nms_all = []
        scores_nms_all = []
        cls_inds = []
        for i_cls in range(S):
            # # prevent out-of-memory during evaluation
            # torch.cuda.empty_cache()

            # NMS based on IOU (or distance)
            ave_lwh_c = ave_lwh[i_cls]
            boxes_c = target_to_boxes_torch(
                reg_pred[bmask, :, i_cls, :], ave_lwh_c, num_theta_bins
            ).view(
                -1, 7
            )  # (N' * A, 7)
            cls_pred_c = cls_pred[bmask, :, i_cls].view(-1)  # (N' * A)
            # nms_inds = ub3d.nms_3d_gpu(boxes_c, cls_pred_c, iou_thresh=0.3)
            nms_inds = ub3d.nms_3d_dist_gpu(
                boxes_c,
                cls_pred_c,
                l_ave=ave_lwh_c[0],
                w_ave=ave_lwh_c[1],
                nms_thresh=0.4,
            )
            boxes_nms = boxes_c[nms_inds].data.cpu().numpy()
            scores_nms = cls_pred_c[nms_inds].data.cpu().numpy()

            # # -------- DEBUG no NMS --------
            # boxes_nms = boxes_i.data.cpu().numpy()
            # scores_nms = cls_pred_i.data.cpu().numpy()
            # # --------

            boxes_nms_all.append(boxes_nms)
            scores_nms_all.append(scores_nms)
            cls_inds.append(i_cls)

        # # -------- DEBUG drawing (batch size needs to be 1) --------
        # from lidar_det.utils.viz_pyviz import draw_lidar

        # viz = _pyviz_render(batch_dict)
        # cls_mapping = batch_dict["params"]["class_mapping"]
        # for s in range(S):
        #     pts_c = (10.0 * scores_nms_all[s]).clip(min=0, max=1)
        #     pts_n = cls_mapping[cls_inds[s]]
        #     viz = draw_lidar(
        #         pc=boxes_nms_all[s][:, :3],
        #         pts_name=f"pred_{pts_n}",
        #         pts_color=(0.0, 1.0, 0.0),
        #         pts_alpha=pts_c,
        #         pts_size=200,
        #         viz=viz,
        #     )

        # viz.save("./logs/DEV_tests/pyviz")
        # # --------

        # store detection results
        if nuscenes:
            eval_dict_b = write_nuscenes_results(
                boxes_nms_all,
                scores_nms_all,
                cls_inds,
                batch_dict,
                i_batch,
                output_dir=output_dir,
                plotting=plotting,
            )
            eval_dict.update(eval_dict_b)
        else:
            write_jrdb_results(
                boxes_nms_all[0],
                scores_nms_all[0],
                batch_dict,
                i_batch,
                output_dir=output_dir,
                plotting=plotting,
            )

    return tb_dict, eval_dict


def model_eval_collate_fn(
    tb_dict_list,
    eval_dict_list,
    output_dir,
    full_eval=False,
    rm_files=False,
    nuscenes=False,
):
    # tb_dict should only contain scalar values, collate them into an array
    # and take their mean as the value of the epoch
    epoch_tb_dict = {}
    for batch_tb_dict in tb_dict_list:
        for k, v in batch_tb_dict.items():
            epoch_tb_dict.setdefault(k, []).append(v)
    for k, v in epoch_tb_dict.items():
        epoch_tb_dict[k] = np.array(v).mean()

    if not full_eval:
        return epoch_tb_dict

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prevent out-of-memory during evaluation
    torch.cuda.empty_cache()

    if nuscenes:
        nusc_sub_dict = {
            "meta": {
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
        }
        results_dict = {}
        for result_dict in eval_dict_list:
            results_dict.update(result_dict)
        nusc_sub_dict["results"] = results_dict

        nusc_json = os.path.join(os.path.join(output_dir, "nusc_sub_dict.json"))
        with open(nusc_json, "w") as f:
            json.dump(nusc_sub_dict, f)

        # NOTE hard-coded datapath
        eval_nuscenes(
            nusc_json,
            output_dir,
            "./data/nuScenes",
            "val",
            "v1.0-trainval",
            # "mini_val",
            # "v1.0-mini",
            plot_examples=10,
            render_curves=True,
            verbose=True,
        )

        with open(os.path.join(output_dir, "metrics_summary.json"), "r") as f:
            eval_results = json.load(f)

        for k, v in eval_results["label_aps"].items():
            epoch_tb_dict[f"ap_{k}_05"] = v["0.5"]
            epoch_tb_dict[f"ap_{k}_10"] = v["1.0"]
            epoch_tb_dict[f"ap_{k}_20"] = v["2.0"]
            epoch_tb_dict[f"ap_{k}_40"] = v["4.0"]

        for k, v in eval_results["mean_dist_aps"].items():
            epoch_tb_dict[f"ap_{k}_ave"] = v

        epoch_tb_dict["ap_all_ave"] = eval_results["mean_ap"]
        epoch_tb_dict["nd_score"] = eval_results["nd_score"]

    else:
        try:
            ap_dict = eval_jrdb(
                gt_dir="./data/JRDB/train_dataset/labels_kitti",
                det_dir=os.path.join(output_dir, "detections"),
                rm_det_files=rm_files,
            )  # NOTE hard-coded path

            for k, v in ap_dict.items():
                epoch_tb_dict[f"ap_{k}"] = v
        except Exception as e:
            print(e)
            print("Evaluation failed")

    # prevent out-of-memory after evaluation
    torch.cuda.empty_cache()

    return epoch_tb_dict


def write_jrdb_results(boxes, scores, batch_dict, i_batch, output_dir, plotting):
    # store detection results
    sequence = batch_dict["sequence"][i_batch]
    det_dir = os.path.join(output_dir, f"detections/{sequence}")
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)

    frame_id = f"{batch_dict['frame_id'][i_batch]:06d}"
    with open(os.path.join(det_dir, f"{frame_id}.txt"), "w") as f:
        f.write(ub3d.boxes_to_string(boxes, scores, jrdb_format=True))

    if plotting:
        boxes_gt = (
            batch_dict["boxes_gt"][i_batch] if "boxes_gt" in batch_dict.keys() else None
        )
        fig, ax = plot_bev(
            batch_dict["points"][i_batch],
            boxes=boxes,
            scores=scores,
            score_thresh=0.3,
            boxes_gt=boxes_gt,
        )

        fig_dir = os.path.join(output_dir, f"figs/{sequence}")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        fig.savefig(os.path.join(fig_dir, f"{frame_id}.png"))

        plt.close(fig)


def write_nuscenes_results(
    boxes_all, scores_all, cls_all, batch_dict, i_batch, output_dir, plotting
):
    # at most 500 boxes can be kept, keep the highest confidence ones
    scores_kept = np.concatenate([s for s in scores_all])
    if len(scores_kept) > 500:
        min_score = np.sort(scores_kept)[-500]
    else:
        min_score = -1

    # convert to nuscenes sample_result
    sr_list = []  # sample result list
    sample_token = batch_dict["sample_token"][i_batch]
    pc_offset = batch_dict["points_offset"][i_batch]
    cls_mapping = batch_dict["params"]["class_mapping"]
    for boxes, scores, cls_idx in zip(boxes_all, scores_all, cls_all):
        # back to the nuScenes global frame
        boxes_raw = boxes[scores > min_score]
        scores = scores[scores > min_score]
        boxes_raw[:, :3] = boxes_raw[:, :3] + pc_offset.T
        detection_name = cls_mapping[cls_idx]
        sr_list += [
            ub3d.box_to_nuscenes(box, score, sample_token, detection_name)
            for box, score in zip(boxes_raw, scores)
        ]

    result_dict = {sample_token: sr_list}

    if plotting:
        if "boxes_gt" in batch_dict.keys():
            boxes_gt = batch_dict["boxes_gt"][i_batch]
            boxes_gt_cls = batch_dict["boxes_gt_cls"][i_batch]
        else:
            boxes_gt = None
            boxes_gt_cls = None

        fig, ax = plot_bev(
            batch_dict["points"][i_batch],
            boxes_gt=boxes_gt,
            boxes_gt_cls=boxes_gt_cls,
            xlim=(-20, 20),
            ylim=(-20, 20),
            # xlim=(5, 20),
            # ylim=(-5, 10),
        )

        for boxes, scores, cls_idx in zip(boxes_all, scores_all, cls_all):
            fig, ax = plot_bev(
                boxes=boxes,
                scores=scores,
                boxes_cls=cls_idx,
                score_thresh=0.3,
                fig=fig,
                ax=ax,
            )

        fig_dir = os.path.join(output_dir, "figs")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        frame_id = f"{batch_dict['frame_id'][i_batch]:06d}"
        fig.savefig(os.path.join(fig_dir, f"{frame_id}.png"))

        plt.close(fig)

    return result_dict


def compute_box_loss_disentangled(pred_boxes, target_boxes, tb_dict, weight=None):
    # pred_boxes (N, 7), target_boxes (N, 7), fg points only
    N = pred_boxes.shape[0]

    target_corners = ub3d.boxes_to_corners_torch(target_boxes)  # (N, 3, 8)

    xyz_gt = target_boxes[:, :3]
    lwh_gt = target_boxes[:, 3:6]
    R_gt = ub3d.get_R_torch(target_boxes)

    xyz_pred = pred_boxes[:, :3]
    lwh_pred = pred_boxes[:, 3:6]
    R_pred = ub3d.get_R_torch(pred_boxes)
    # theta_pred = pred_boxes[6]

    c_xyz = torch.tensor(
        [
            [1, 1, -1, -1, 1, 1, -1, -1],
            [-1, 1, 1, -1, -1, 1, 1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
        ],
        device=pred_boxes.device,
    ).float()  # (3, 8)
    c_xyz = 0.5 * c_xyz.unsqueeze(dim=0)  # (1, 3, 8)

    # loc loss
    pred_corners_loc = R_gt @ (c_xyz * lwh_gt.unsqueeze(dim=2)) + xyz_pred.unsqueeze(
        dim=2
    )
    loc_loss = F.mse_loss(pred_corners_loc.view(N, -1), target_corners.view(N, -1))
    tb_dict["loc_loss"] = loc_loss.item()

    # dim loss
    pred_corners_dim = R_gt @ (c_xyz * lwh_pred.unsqueeze(dim=2)) + xyz_gt.unsqueeze(
        dim=2
    )
    lwh_loss = F.mse_loss(pred_corners_dim.view(N, -1), target_corners.view(N, -1))
    tb_dict["lwh_loss"] = lwh_loss.item()

    # ori loss
    pred_corners_pred = R_pred @ (c_xyz * lwh_gt.unsqueeze(dim=2)) + xyz_gt.unsqueeze(
        dim=2
    )
    ori_loss = F.mse_loss(pred_corners_pred.view(N, -1), target_corners.view(N, -1))
    tb_dict["ori_loss"] = ori_loss.item()

    box_loss = loc_loss + lwh_loss + ori_loss

    return box_loss, tb_dict
