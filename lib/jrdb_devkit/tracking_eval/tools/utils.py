"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import pandas as pd
import numpy as np

from .mot import MOTAccumulator
from .distances import iou_matrix, norm2squared_matrix, iou_matrix_3d


def compare_to_groundtruth(gt, dt, dist='iou', distfields=['X', 'Y', 'Width', 'Height'], distth=0.5, _3d = False):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids
    
    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results
    
    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """


    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)
    
    def compute_euc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)
    
    def compute_3d_iou(a,b):
        return iou_matrix_3d(a, b, max_iou = distth)

    if _3d:
        compute_dist = compute_3d_iou if dist.upper() == 'IOU' else compute_euc
    else:
        compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulator()

    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in 
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]
    
    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0,0))

        if fid in gt.index:
            fgt = gt.loc[fid] 
            oids = fgt.index.values

        if fid in dt.index:
            fdt = dt.loc[fid]
            hids = fdt.index.values

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(fgt[distfields].values, fdt[distfields].values)
        if _3d:
            if oids.shape[0] > 0:
                gt_dists = np.sqrt(norm2squared_matrix(fgt[distfields], np.zeros((1, fgt[distfields].values.shape[1]))))
            else:
                gt_dists = None
            if hids.shape[0] > 0:
                det_dists = np.sqrt(norm2squared_matrix(fdt[distfields], np.zeros((1, fdt[distfields].values.shape[1]))))
            else:
                det_dists = None
        else:
            gt_dists = None
            det_dists = None
        acc.update(oids, hids, dists, frameid=fid, gt_dists=gt_dists, det_dists=det_dists)
    
    return acc
