from mayavi import mlab
import numpy as np
from PIL import Image

import lidar_det.utils.utils_box3d as ub3d
from lidar_det.detector import PersonMinkUNet

# # https://docs.enthought.com/mayavi/mayavi/tips.html#rendering-using-the-virtual-framebuffer
# from pyvirtualdisplay.display import Display
# display = Display(visible=True, size=(1280, 1024))
# display.start()

mlab.options.offscreen = True

ckpt = "/globalwork/jia/archive/JRDB_cvpr21_workshop/" \
    "logs/unet_bl_voxel_jrdb_0.05_0.1_20210519_232859/ckpt/ckpt_e40.pth"
detector = PersonMinkUNet(ckpt)


def draw_pointcloud_and_detections(
    pc: np.ndarray, boxes: np.ndarray, scores: np.ndarray
):
    fig = mlab.figure(
        bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), engine=None, size=(800, 500),
    )

    # points
    s = np.hypot(pc[0], pc[1])
    # print(pc.mean(axis=1))
    mpt = mlab.points3d(
        pc[0],
        pc[1],
        pc[2],
        s,
        colormap="blue-red",
        mode="sphere",
        scale_factor=0.06,
        figure=fig,
    )
    mpt.glyph.scale_mode = "scale_by_vector"

    # boxes
    s_argsort = scores.argsort()
    scores = scores[s_argsort]
    # plot low confidence boxes first (on bottom layer)
    boxes = boxes[s_argsort]
    boxes_color = [(0.0, 1.0, 0.0)] * len(boxes)

    # limit the number of boxes so it does not take forever to plot
    if len(boxes) > 50:
        boxes = boxes[-50:]
        boxes_color = boxes_color[-50:]

    corners, connect_inds = ub3d.boxes_to_corners(boxes, connect_inds=True)
    for corner, color in zip(corners, boxes_color):
        for inds in connect_inds:
            mlab.plot3d(
                corner[0, inds],
                corner[1, inds],
                corner[2, inds],
                color=color,
                tube_radius=None,
                line_width=1.0,
                figure=fig,
            )

    # print(mlab.view())

    mlab.view(
        # azimuth=180,
        # elevation=180,
        # focalpoint=[12.0909996, -1.04700089, -2.03249991],
        # distance="auto",
        focalpoint=pc.mean(axis=1),
        distance=20,
        figure=fig,
    )

    # mlab.show()  # for finding a good view interactively

    # convert to image
    fig.scene._lift()
    img = mlab.screenshot(figure=fig)
    mlab.close(fig)

    return Image.fromarray(img)


def run_detector_plot_result(pc: np.ndarray, score_threshold: float) -> Image.Image:
    boxes, scores = detector(pc)  # (B 7), (B)
    mask = scores >= score_threshold
    img = draw_pointcloud_and_detections(pc, boxes[mask], scores[mask])
    return img


if __name__ == "__main__":
    import yaml
    from lidar_det.dataset import JRDBDet3D
    # import matplotlib.pyplot as plt

    cfg_file = "/globalwork/jia/archive/JRDB_cvpr21_workshop/logs/" \
        "unet_bl_voxel_jrdb_0.05_0.1_20210519_232859/backup/" \
        "unet_bl_voxel_jrdb_0.05_0.1.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["target_mode"] = cfg["model"]["target_mode"]
    cfg["dataset"]["num_anchors"] = cfg["model"]["kwargs"]["num_anchors"]
    cfg["dataset"]["num_ori_bins"] = cfg["model"]["kwargs"]["num_ori_bins"]

    pc_path = "/globalwork/datasets/JRDB_may17"
    dataset = JRDBDet3D(pc_path, "test", cfg["dataset"])

    # pc = dataset[0]["points"]
    # img = run_detector_plot_result(pc, 0.5)

    # # plot plt
    # fig = plt.figure(figsize=(7, 5))
    # plt.imshow(img)
    # plt.show()
