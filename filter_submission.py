# remove low confidence detections from the submission file

import os
import numpy as np

THRESH = 1e-3
SUBMISSION_DIR = "/home/jia/jrdb_person_minkunet"
OUT_DIR = "/home/jia/jrdb_person_minkunet_new"

paths = os.listdir(SUBMISSION_DIR)
for i, path in enumerate(paths):
    print(f"process [{i}/{len(paths)}] {path}")
    seq_path = os.path.join(SUBMISSION_DIR, path)
    seq_path_new = os.path.join(OUT_DIR, path)
    os.makedirs(seq_path_new, exist_ok=True)

    for file in os.listdir(seq_path):
        det_path = os.path.join(seq_path, file)
        with open(det_path, "r") as f:
            lines = f.readlines()
        confs = np.asarray([float(line.strip("\n").split(" ")[-1]) for line in lines])
        lines_new = np.asarray(lines)[confs > THRESH]
        # print(len(lines), len(lines_new))

        det_path_new = os.path.join(seq_path_new, file)
        with open(det_path_new, "w") as f:
            f.writelines(lines_new.tolist())

