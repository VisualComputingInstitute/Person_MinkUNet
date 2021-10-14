# Should only be used by code under ./tests


def get_cfgs():
    model_cfg = {
        "type": "MinkUNet",
        # "type": "MinkResNet",
        "kwargs": {
            "cr": 1.0,
            "run_up": True,
            "num_anchors": 1,
            "num_ori_bins": 12,
            # "fpn": True
            "fpn": False,
        },
        "target_mode": 2,  # 0-inclusion, 1-anchor, 2-dist, 3-dist and reweighting
        "disentangled_loss": False,
    }

    dataset_cfg = {
        "name": "JRDB",
        # "name": "nuScenes",
        "augmentation": True,
        # "voxel_size": [0.05, 0.05, 0.05],
        "voxel_size": [0.05, 0.05, 0.20],
        "num_points": 1e6,
        "canonical": False,
        "dist_min": 0.5,
        "dist_max": 1.0,
        # "included_classes": ["pedestrian"],
        "included_classes": [],
        "nsweeps": 1,
        "additional_features": []
    }

    dataset_cfg["target_mode"] = model_cfg["target_mode"]
    dataset_cfg["num_anchors"] = model_cfg["kwargs"]["num_anchors"]
    dataset_cfg["num_ori_bins"] = model_cfg["kwargs"]["num_ori_bins"]
    if dataset_cfg["name"] == "nuScenes":
        nc = len(dataset_cfg["included_classes"])
        model_cfg["kwargs"]["num_classes"] = nc if nc > 0 else 10
        model_cfg["nuscenes"] = True
        model_cfg["kwargs"]["input_dim"] = 3 + len(dataset_cfg["additional_features"])
    else:
        model_cfg["kwargs"]["num_classes"] = 1
        model_cfg["nuscenes"] = False
        model_cfg["kwargs"]["input_dim"] = 3

    return model_cfg, dataset_cfg
