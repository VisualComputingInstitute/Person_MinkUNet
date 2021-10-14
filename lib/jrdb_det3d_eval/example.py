from jrdb_det3d_eval import eval_jrdb


det_dir = "/globalwork/jia/dr_spaam_output/logs/20201210_124150_jrdb_det_3d_EVAL/output/val/e000000/detections"
# gt_dir = "/globalwork/jia/jrdb_eval/val"
gt_dir = "/globalwork/datasets/JRDB/train_dataset/labels_kitti"

ap_dict = eval_jrdb(gt_dir, det_dir)
for k, v in ap_dict.items():
    print(k, v)
