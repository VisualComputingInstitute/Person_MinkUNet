# RUN="unet_bl_voxel_jrdb_0.05_0.1_20210518_220414"
RUN="unet_bl_voxel_jrdb_0.05_0.1_20210519_232859"

python bin/train.py \
--cfg /globalwork/jia/archive/JRDB_cvpr21_workshop/logs/$RUN/backup/unet_bl_voxel_jrdb_0.05_0.1.yaml \
--ckpt /globalwork/jia/archive/JRDB_cvpr21_workshop/logs/$RUN/ckpt/ckpt_e40.pth \
--evaluation \
--tmp \
--bs_one