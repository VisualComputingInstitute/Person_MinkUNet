import numpy as np
import matplotlib.pyplot as plt

from pyviz3d.visualizer import Visualizer


def draw_lidar(
    pc=None,
    pts_name="points",
    pts_color=None,
    pts_alpha=None,
    pts_show=False,
    pts_size=25,
    viz=None,
):
    # pc (N, 3), boxes (B, 7)
    if viz is None:
        viz = Visualizer()

    if pc is not None:
        if pts_color is None:
            s = np.hypot(pc[:, 0], pc[:, 1])
            pts_color = plt.cm.jet(np.clip(s / 30, 0, 1))[:, :3]

        if isinstance(pts_color, tuple):
            pts_color = np.array(pts_color).reshape(1, 3).repeat(pc.shape[0], axis=0)
        elif len(pts_color.shape) == 1:
            pts_color = plt.cm.jet(pts_color)[:, :3]

        if pts_alpha is not None:
            if isinstance(pts_alpha, np.ndarray):
                pts_alpha = pts_alpha.reshape(pts_color.shape[0], 1)
            pts_color = pts_alpha * pts_color + (1.0 - pts_alpha) * np.ones_like(
                pts_color
            )

        if pts_color.max() <= 1.001:
            pts_color = (pts_color * 255).astype(np.uint8)

        viz.add_points(
            pts_name, pc, colors=pts_color, visible=pts_show, point_size=pts_size,
        )

    # viz.save()

    return viz
