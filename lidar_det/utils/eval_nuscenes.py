import os

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval


def eval_nuscenes(
    result_json,
    out_dir,
    data_root,
    split,
    version,
    plot_examples=10,
    render_curves=True,
    verbose=True,
):
    # Ref https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py  # noqa
    result_json = os.path.expanduser(result_json)
    out_dir = os.path.expanduser(out_dir)
    data_root = os.path.expanduser(data_root)
    cfg = config_factory("detection_cvpr_2019")

    nusc = NuScenes(version=version, verbose=verbose, dataroot=data_root)
    nusc_eval = DetectionEval(
        nusc,
        config=cfg,
        result_path=result_json,
        eval_set=split,
        output_dir=out_dir,
        verbose=verbose,
    )
    nusc_eval.main(plot_examples=plot_examples, render_curves=render_curves)
