from functools import partial

from .nets.minkunet import MinkUNet
from .nets.minkunet_pillar import MinkPillarUNet
from .nets.minkresnet import MinkResNet

__all__ = [
    "get_model",
    "MinkUNetDetector",
    "MinkPillarUNetDetector",
    "MinkResNetDetector",
]


def get_model(model_cfg, inference_only=False):
    if model_cfg["type"] == "MinkUNet":
        net = MinkUNetDetector(**model_cfg["kwargs"])
    elif model_cfg["type"] == "MinkPillarUNet":
        net = MinkPillarUNetDetector(**model_cfg["kwargs"])
    elif model_cfg["type"] == "MinkResNet":
        net = MinkResNetDetector(**model_cfg["kwargs"])
    else:
        raise RuntimeError(f"Unknown model '{model_cfg['type']}'")

    if not inference_only:
        from .model_fn import model_fn, model_eval_fn, model_eval_collate_fn, error_fn

        net.model_fn = partial(
            model_fn,
            target_mode=model_cfg["target_mode"],
            disentangled_loss=model_cfg["disentangled_loss"],
        )
        net.model_eval_fn = partial(model_eval_fn, nuscenes=model_cfg["nuscenes"])
        net.model_eval_collate_fn = partial(
            model_eval_collate_fn, nuscenes=model_cfg["nuscenes"]
        )
        net.error_fn = error_fn

    return net


class MinkUNetDetector(MinkUNet):
    def __init__(
        self,
        num_anchors,
        num_ori_bins,
        cr=1.0,
        run_up=True,
        fpn=False,
        num_classes=1,
        input_dim=3,
    ):
        out_dim = _get_num_output_channels(num_ori_bins, num_anchors, num_classes)
        super().__init__(cr=cr, run_up=run_up, num_classes=out_dim, input_dim=input_dim)
        self._na = num_anchors
        self._no = num_ori_bins
        self._nc = num_classes

    @property
    def num_anchors(self):
        return self._na

    @property
    def num_classes(self):
        return self._nc


class MinkPillarUNetDetector(MinkPillarUNet):
    def __init__(
        self,
        num_anchors,
        num_ori_bins,
        cr=1.0,
        run_up=True,
        fpn=False,
        num_classes=1,
        input_dim=3,
    ):
        out_dim = _get_num_output_channels(num_ori_bins, num_anchors, num_classes)
        super().__init__(cr=cr, run_up=run_up, num_classes=out_dim, input_dim=input_dim)
        self._na = num_anchors
        self._no = num_ori_bins
        self._nc = num_classes

    @property
    def num_anchors(self):
        return self._na

    @property
    def num_classes(self):
        return self._nc


class MinkResNetDetector(MinkResNet):
    def __init__(
        self,
        num_anchors,
        num_ori_bins,
        cr=1.0,
        run_up=True,
        fpn=False,
        num_classes=1,
        input_dim=3,
    ):
        out_dim = _get_num_output_channels(num_ori_bins, num_anchors, num_classes)
        super().__init__(cr=cr, num_classes=out_dim, fpn=fpn, input_dim=input_dim)
        self._na = num_anchors
        self._no = num_ori_bins
        self._nc = num_classes

    @property
    def num_anchors(self):
        return self._na

    @property
    def num_classes(self):
        return self._nc


def _get_num_output_channels(num_ori_bins, num_anchors, num_classes):
    if num_ori_bins > 1:
        # out_dim = num_anchors * (num_ori_bins + 8)
        out_dim = num_anchors * (2 * num_ori_bins + 7)
    else:
        out_dim = num_anchors * 8

    out_dim *= num_classes

    return out_dim
