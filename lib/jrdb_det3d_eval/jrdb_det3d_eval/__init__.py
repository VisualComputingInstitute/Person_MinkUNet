import os
import warnings
import numpy as np
import shutil

from .kitti_common import get_label_annos
from .eval import get_official_eval_result


def _eval_seq(gt_annos, det_annos):
    with warnings.catch_warnings():
        # so numba compile warning does not pollute logs
        warnings.simplefilter("ignore")
        result_str, result_dict = get_official_eval_result(gt_annos, det_annos, 1)

    # print(result_str)
    # for k, v in result_dict.items():
    #     print(k, v)
    seq_ap = result_dict["Pedestrian_3d/moderate_R40"]

    return seq_ap


def eval_jrdb(gt_dir, det_dir, rm_det_files=False):
    # gt_sequences = sorted(os.listdir(gt_dir))
    det_sequences = sorted(os.listdir(det_dir))
    # assert gt_sequences == det_sequences

    ap_dict = {}

    # per sequence eval
    seq_ap, seq_len = [], []
    for idx, seq in enumerate(det_sequences):
        print(f"({idx + 1}/{len(det_sequences)}) Evaluating {seq}")

        gt_annos = get_label_annos(os.path.join(gt_dir, seq))
        det_annos = get_label_annos(os.path.join(det_dir, seq))

        ap_dict[seq] = _eval_seq(gt_annos, det_annos)
        print(f"{seq}, AP={ap_dict[seq]:.4f}, len={len(gt_annos)}")

        seq_ap.append(ap_dict[seq])
        seq_len.append(len(gt_annos))

    # NOTE Jointly evaluating all sequences crashes, don't know why. Use average
    # AP of all sequences instead.
    print("Evaluating whole set")
    seq_ap = np.array(seq_ap)
    seq_len = np.array(seq_len)
    ap_dict["all"] = np.sum(seq_ap * (seq_len / seq_len.sum()))
    print(f"Whole set, AP={ap_dict['all']:.4f}, len={seq_len.sum()}")

    if rm_det_files:
        shutil.rmtree(det_dir)

    return ap_dict
