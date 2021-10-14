import numpy as np
from mayavi import mlab

import lidar_det.utils.utils_box3d as ub3d


def draw_lidar(
    pts=None,
    pts_color=None,
    pts_scale=0.06,
    pts_mode="sphere",
    fig=None,
    fig_size=(800, 500),
    boxes=None,
    boxes_color=(0.0, 1.0, 0.0),
    scores=None,
    score_thresh=0.0,
    boxes_gt=None,
    boxes_gt_color=(1.0, 0.0, 0.0),
    color_bar=False
):
    """Draw LiDAR points

    Args:
        pts (array[3, N]): xyz
        pts_color (array[N, 3] or tuple(3))
        boxes (array[B, 7])
        scores (array[B]): Used to color code box
        score_threh: Box with lower scores are not plotted
        boxes_gt (array[B, 7])

    Returns:
        fig
    """
    if fig is None:
        fig = mlab.figure(
            figure=None,
            bgcolor=(1, 1, 1),
            fgcolor=(0, 0, 0),
            engine=None,
            size=fig_size,
        )

    if pts is not None:
        if pts_color is None:
            s = np.hypot(pts[0], pts[1])
            mpt = mlab.points3d(
                pts[0],
                pts[1],
                pts[2],
                s,
                colormap="blue-red",
                mode=pts_mode,
                scale_factor=pts_scale,
                figure=fig,
            )
            mpt.glyph.scale_mode = "scale_by_vector"
            if color_bar:
                mlab.scalarbar(
                    object=mpt, nb_labels=8, label_fmt="%.1f", orientation="vertical"
                )
        else:
            mlab.points3d(
                pts[0],
                pts[1],
                pts[2],
                color=pts_color,
                mode=pts_mode,
                scale_factor=pts_scale,
                figure=fig,
            )

    # boxes
    if boxes is not None and len(boxes) > 0:
        if scores is not None:
            boxes = boxes[scores >= score_thresh]
            scores = scores[scores >= score_thresh]
            # plot low confidence boxes first (on bottom layer)
            s_argsort = scores.argsort()
            boxes = boxes[s_argsort]
            scores = scores[s_argsort]
            # boxes_color = plt.cm.Greens(scores)  # TODO
            boxes_color = [(0.0, 1.0, 0.0)] * len(boxes)
        else:
            boxes_color = [boxes_color] * len(boxes)

        # boxes
        corners, connect_inds = ub3d.boxes_to_corners(boxes, connect_inds=True)
        for corner, color in zip(corners, boxes_color):
            for inds in connect_inds:
                mlab.plot3d(
                    corner[0, inds],
                    corner[1, inds],
                    corner[2, inds],
                    color=color,
                    tube_radius=None,
                    line_width=7.0,
                    figure=fig,
                )

    # gt boxes
    if boxes_gt is not None and len(boxes_gt) > 0:
        corners, connect_inds = ub3d.boxes_to_corners(boxes_gt, connect_inds=True)
        for corner in corners:
            for inds in connect_inds:
                mlab.plot3d(
                    corner[0, inds],
                    corner[1, inds],
                    corner[2, inds],
                    color=boxes_gt_color,
                    tube_radius=None,
                    line_width=7.0,
                    figure=fig,
                )

    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62,
        figure=fig,
    )

    return fig
