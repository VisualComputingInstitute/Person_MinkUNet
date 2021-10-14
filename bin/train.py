import argparse
import yaml
import torch

from lidar_det.dataset import get_dataloader
from lidar_det.pipeline import Pipeline
from lidar_det.model import get_model


def run_training(model, pipeline, cfg):
    # main train loop
    train_loader = get_dataloader(
        split="train", shuffle=True, dataset_cfg=cfg["dataset"], **cfg["dataloader"]
    )
    val_loader = get_dataloader(
        split="val", shuffle=False, dataset_cfg=cfg["dataset"], **cfg["dataloader"]
    )
    status = pipeline.train(model, train_loader, val_loader)

    # # test after training
    # if not status:
    #     test_loader = get_dataloader(
    #         split="val",
    #         shuffle=False,
    #         dataset_cfg=cfg["dataset"],
    #         num_workers=1,
    #         batch_size=1,
    #     )
    #     pipeline.evaluate(model, test_loader, tb_prefix="VAL")


def run_evaluation(model, pipeline, cfg):
    val_loader = get_dataloader(
        split="val",
        shuffle=False,
        dataset_cfg=cfg["dataset"],
        num_workers=1,
        batch_size=1,
    )
    pipeline.evaluate(model, val_loader, tb_prefix="VAL", rm_files=False)

    # test_loader = get_dataloader(
    #     split="test",
    #     shuffle=False,
    #     dataset_cfg=cfg["dataset"],
    #     num_workers=1,
    #     batch_size=1,
    # )
    # pipeline.evaluate(model, test_loader, tb_prefix="TEST", rm_files=False)


if __name__ == "__main__":
    # Run benchmark to select fastest implementation of ops.
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg", type=str, required=True, help="configuration of the experiment"
    )
    parser.add_argument("--ckpt", type=str, required=False, default=None)
    parser.add_argument("--cont", default=False, action="store_true")
    parser.add_argument("--tmp", default=False, action="store_true")
    parser.add_argument("--bs_one", default=False, action="store_true")
    parser.add_argument("--evaluation", default=False, action="store_true")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["pipeline"]["Logger"]["backup_list"].append(args.cfg)
        if args.evaluation:
            cfg["pipeline"]["Logger"]["tag"] += "_EVAL"
        if args.tmp:
            cfg["pipeline"]["Logger"]["tag"] = "TMP_" + cfg["pipeline"]["Logger"]["tag"]
        if args.bs_one:
            cfg["dataloader"]["batch_size"] = 1

    cfg["dataset"]["target_mode"] = cfg["model"]["target_mode"]
    cfg["dataset"]["num_anchors"] = cfg["model"]["kwargs"]["num_anchors"]
    cfg["dataset"]["num_ori_bins"] = cfg["model"]["kwargs"]["num_ori_bins"]
    if cfg["dataset"]["name"] == "nuScenes":
        nc = len(cfg["dataset"]["included_classes"])
        cfg["model"]["kwargs"]["num_classes"] = nc if nc > 0 else 10
        cfg["model"]["nuscenes"] = True
        cfg["model"]["kwargs"]["input_dim"] = 3 + len(
            cfg["dataset"]["additional_features"]
        )
    else:
        cfg["model"]["kwargs"]["num_classes"] = 1
        cfg["model"]["nuscenes"] = False
        cfg["model"]["kwargs"]["input_dim"] = 3

    model = get_model(cfg["model"])
    model.cuda()

    pipeline = Pipeline(model, cfg["pipeline"])

    if args.ckpt:
        pipeline.load_ckpt(model, args.ckpt)
    elif args.cont and pipeline.sigterm_ckpt_exists():
        pipeline.load_sigterm_ckpt(model)

    # training or evaluation
    if not args.evaluation:
        run_training(model, pipeline, cfg)
    else:
        run_evaluation(model, pipeline, cfg)

    pipeline.close()
