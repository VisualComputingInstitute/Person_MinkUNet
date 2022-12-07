# Person-MinkUNet

PyTorch implementation of Person-MinkUNet.
Winner of JRDB 3D detection challenge in JRDB-ACT Workshop at CVPR 2021
[`[arXiv]`](https://arxiv.org/abs/2107.06780)
[`[video]`](https://www.youtube.com/watch?v=RnGnONoX9cU)
[`[leaderboard]`](https://jrdb.erc.monash.edu/leaderboards/detection).

# Prerequisite

- `python>=3.8`
- `torchsparse==1.2.0` [(link)](https://github.com/mit-han-lab/torchsparse)
- `PyTorch==1.6.0`

# Quick start

Download [JackRabbot dataset](https://jrdb.stanford.edu/) under `PROJECT/data/JRDB`.

```
# install lidar_det project
python setup.py develop

# build libraries
cd lib/iou3d
python setup.py develop

cd ../jrdb_det3d_eval
python setup.py develop
```

Run 
```
python bin/train.py --cfg PATH_TO_CFG [--ckpt PATH_TO_CKPT] [--evaluation]
```

# Model zoo

| Split | Checkpoint | Config |
|-------|------------|--------|
| train       | [ckpt](https://github.com/VisualComputingInstitute/Person_MinkUNet/releases/download/v1.0/ckpt_e40_train.pth) | [cfg](https://github.com/VisualComputingInstitute/Person_MinkUNet/releases/download/v1.0/unet_bl_voxel_jrdb_0.05_0.1.yaml) |
| train + val | [ckpt](https://github.com/VisualComputingInstitute/Person_MinkUNet/releases/download/v1.0/ckpt_e40_train_val.pth) | [cfg](https://github.com/VisualComputingInstitute/Person_MinkUNet/releases/download/v1.0/unet_bl_voxel_jrdb_0.05_0.1.yaml) |

# Acknowledgement

- torchsparse [(link)](https://github.com/mit-han-lab/torchsparse)
- PointRCNN [(link)](https://github.com/sshaoshuai/PointRCNN/tree/master/lib/utils/iou3d)

# Citation
```
@inproceedings{Jia2021PersonMinkUnet,
  title        = {{Person-MinkUNet: 3D Person Detection with LiDAR Point Cloud}},
  author       = {Dan Jia and Bastian Leibe},
  booktitle    = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year         = {2021}
}
```





