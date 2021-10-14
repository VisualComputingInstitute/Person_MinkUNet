import numpy as np
import matplotlib.pyplot as plt

import lidar_det.utils.utils_box3d as ub3d


def plot_bev(
    pc=None,
    pts_color=None,
    pts_scale=0.01,
    title=None,
    fig=None,
    ax=None,
    xlim=(-10, 10),
    ylim=(-10, 10),
    boxes=None,
    boxes_cls=None,
    scores=None,
    score_thresh=0.0,
    boxes_gt=None,
    boxes_gt_cls=None,
):
    """Plot BEV of LiDAR points

    Args:
        pc (array[3, N]): xyz
        pts_color (array[N, 3] or tuple(3))
        boxes (array[B, 7])
        scores (array[B]): Used to color code box
        score_threh: Box with lower scores are not plotted
        boxes_gt (array[B, 7])

    Returns:
        fig, ax
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

        if title is not None:
            ax.set_title(f"{title}")

    # lidar
    if pc is not None:
        if pts_color is None:
            s = np.hypot(pc[0], pc[1])
            pts_color = plt.cm.jet(np.clip(s / 30, 0, 1))

        ax.scatter(pc[0], pc[1], color=pts_color, s=pts_scale)

    # boxes
    if boxes is not None and len(boxes) > 0:
        if scores is not None:
            boxes = boxes[scores >= score_thresh]
            scores = scores[scores >= score_thresh]
            # plot low confidence boxes first (on bottom layer)
            s_argsort = scores.argsort()
            boxes = boxes[s_argsort]
            scores = scores[s_argsort]

        # color coded classes
        boxes_color = get_boxes_color(boxes, boxes_cls, (0.0, 1.0, 0.0), scores)
        corners = ub3d.boxes_to_corners(boxes)
        for corner, c in zip(corners, boxes_color):
            inds = [0, 3, 2, 1]
            ax.plot(corner[0, inds], corner[1, inds], linestyle="-", color=c)
            ax.plot(corner[0, :2], corner[1, :2], linestyle="--", color=c)

    if boxes_gt is not None and len(boxes_gt) > 0:
        boxes_gt_color = get_boxes_color(boxes_gt, boxes_gt_cls, (1.0, 0.0, 0.0))
        corners = ub3d.boxes_to_corners(boxes_gt)
        for corner, c in zip(corners, boxes_gt_color):
            inds = [0, 3, 2, 1]
            ax.plot(
                corner[0, inds],
                corner[1, inds],
                linestyle="dotted",
                color=c,
                linewidth=2.0,
            )
            ax.plot(
                corner[0, :2], corner[1, :2], linestyle="--", color=c, linewidth=2.0
            )

    return fig, ax


def get_boxes_color(boxes, boxes_cls, default_color, alphas=None):
    B = len(boxes)
    if boxes_cls is not None:
        if isinstance(boxes_cls, (int, float)):
            boxes_cls = boxes_cls * np.ones(B)
        boxes_color = plt.cm.prism(boxes_cls / 10)
        if alphas is not None:
            boxes_color[:, 3] = alphas
    else:
        boxes_color = np.tile(np.array(default_color), (B, 1))
        if alphas is not None:
            boxes_color = np.concatenate((boxes_color, alphas.reshape(B, 1)), axis=1)

    return boxes_color
