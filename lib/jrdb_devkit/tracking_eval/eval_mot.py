"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics

Files
-----
All file content, ground truth, and test files have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for 2D ground truth data:
    <GT_ROOT>/tracking/<SEQUENCE_1>/<IMAGE_NUM>/gt.txt (e.g. tracking/cubberly-auditorium-2019-04-22_1/image_stitched/gt.txt)
    <GT_ROOT>/tracking/<SEQUENCE_2>/<IMAGE_NUM>/gt.txt (e.g. tracking/discovery-walk-2019-02-28_0/image_2/gt/gt.txt)
    ...

Expected 2D directory structure:
    <TEST_ROOT>/<SEQUENCE_1_IMAGE_NUM>.txt (e.g. cubberly-auditorium-2019-04-22_1_image_0.txt)
    <TEST_ROOT>/<SEQUENCE_2_IMAGE_NUM>.txt (e.g. discovery-walk-2019-02-28_0_image_2.txt)
    ...

Layout for 3D ground truth data:
    <GT_ROOT>/tracking/<SEQUENCE_1>/pointclouds/3d_gt.txt (e.g. tracking/cubberly-auditorium-2019-04-22_1/pointclouds/3d_gt.txt)
    <GT_ROOT>/tracking/<SEQUENCE_2>/pointclouds/3d_gt.txt (e.g. tracking/discovery-walk-2019-02-28_0/pointclouds/3d_gt.txt)
    ...

Expected 3D directory structure:
    <TEST_ROOT>/<SEQUENCE_1>.txt (e.g. cubberly-auditorium-2019-04-22_1.txt)
    <TEST_ROOT>/<SEQUENCE_2>.txt (e.g. discovery-walk-2019-02-28_0.txt)
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.
"""

import argparse
import glob
import os, pdb
import logging
import pandas as pd
from collections import OrderedDict
from pathlib import Path

# Handles loading the tools library from the server and as a standalone script.
try:
    from . import tools as mm
except:
    import tools as mm

def compare_dataframes(gts, ts, _3d = False):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing {}...'.format(k))
            if _3d:
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distfields=['X', 'Y', 'Z',
                'Length', 'Height', 'Width', 'Theta'], distth=0.7, _3d = _3d))
            else:
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5, _3d = _3d))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

def get_cameras(tsfiles):
    images = []
    for f in tsfiles:
        image = ('_').join(f.split('/')[-1].split('.')[0].split('_')[-2:])
        if image not in images:
            images.append(image)

    return images

def downselect_gt(cameras, gtfiles):
    new_gts = []
    for g in gtfiles:
        gt_cam = Path(g).parts[-2]
        if gt_cam == 'gt':
            gt_cam = 'image_stitched'
        else:
            gt_cam = gt_cam[3:]
        if gt_cam in cameras:
            new_gts.append(g)

    return new_gts

def evaluate(groundtruths, tests, loglevel='info', fmt='mot15-2D', depth=False,
             solver=None):
    loglevel = getattr(logging, loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if solver:
        mm.lap_util.default_solver = solver

    if not depth:
        gtfiles = glob.glob(os.path.join(groundtruths,'*/gt/gt.txt'))
        tsfiles = [f for f in glob.glob(os.path.join(tests, '*.txt')) if("3d.txt" not in f and not os.path.basename(f).startswith('eval') and not 'result' in os.path.basename(f))]
        cameras = get_cameras(tsfiles)
        gtfiles = downselect_gt(cameras, gtfiles)
    else:
        gtfiles = glob.glob(os.path.join(groundtruths, '*/gt/3d_gt.txt'))
        tsfiles = [f.replace('_3d','') for f in glob.glob(os.path.join(tests, '*.txt')) if(not os.path.basename(f).startswith('eval') and not 'result' in os.path.basename(f))]
 
    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap_util.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap_util.default_solver))
    logging.info('Loading files.')

    if not depth:
        gt_str = ', '.join(sorted(['_'.join(Path(f).parts[-3:-1]).replace('gt_','').replace('gt','image_stitched')+'.txt' for f in gtfiles]))
    else:
        gt_str = ', '.join(sorted([Path(f).parts[-3]+'.txt' for f in gtfiles]))
    ts_str = ', '.join(sorted([Path(f).parts[-1] for f in tsfiles]))

    if len(gtfiles) != len(tsfiles):
        return "Error: There are " + str(len(gtfiles)) + " sequences but only " + str(len(tsfiles)) + " were provided. Please ensure that all sequences are provided. If there are 0 provided, a common error is not ensuring the sequences are directly under the root in the zipped file. If there are 0 groundtruth files, that means the submission files were not named properly to indicate the camera (and did not, for example, end with _image_stitched.txt).<br><br>The list of gt sequences is: " + gt_str + "<br><br>The list of provided sequences is: " + ts_str + ".<br><br><br><br>Feel free to reach out to jrdb@cs.stanford.edu for assistance."
    elif gt_str != ts_str:
        return "Error: The sequences are not correctly named.<br><br>The list of gt sequences is: " + gt_str + ".<br><br>The list of provided sequences is: " + ts_str + ".<br><br><br><br>Feel free to reach out to jrdb@cs.stanford.edu for assistance."
        
    if not depth:
        gt = OrderedDict([(('_').join(Path(f).parts[-3:-1]).replace('gt_','').replace('gt','image_stitched'), mm.io.loadtxt(f, fmt=fmt, min_confidence=-1, _3d=depth)) for f in gtfiles])
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=fmt, _3d=depth)) for f in tsfiles ])
    else:
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=fmt, min_confidence=-1, _3d=depth)) for f in gtfiles])
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=fmt, _3d=depth)) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts, _3d=depth)

    indoor_outdoor_split = \
        {'INDOOR': ['cubberly-auditorium-2019-04-22_1', 'gates-ai-lab-2019-04-17_0',
                    'gates-basement-elevators-2019-01-17_0',
                    'nvidia-aud-2019-01-25_0', 'nvidia-aud-2019-04-18_1',
                    'nvidia-aud-2019-04-18_2', 'gates-foyer-2019-01-17_0',
                    'stlc-111-2019-04-19_1', 'stlc-111-2019-04-19_2',
                    'hewlett-class-2019-01-23_0', 'hewlett-class-2019-01-23_1',
                    'huang-2-2019-01-25_1', 'indoor-coupa-cafe-2019-02-06_0',
                    'tressider-2019-04-26_0', 'tressider-2019-04-26_1',
                    'tressider-2019-04-26_3'],
        'OUTDOOR': ['discovery-walk-2019-02-28_0', 'meyer-green-2019-03-16_1',
                    'discovery-walk-2019-02-28_1', 'food-trucks-2019-02-12_0',
                    'quarry-road-2019-02-28_0', 'outdoor-coupa-cafe-2019-02-06_0',
                    'serra-street-2019-01-30_0', 'gates-to-clark-2019-02-28_0',
                    'tressider-2019-03-16_2', 'lomita-serra-intersection-2019-01-30_0',
                    'huang-intersection-2019-01-22_0']}
    logging.info('Running metrics')
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics,
                              generate_overall=True, _3d=depth, sequence_splits=indoor_outdoor_split,
                              dist_cutoffs=[5, 10, 15, 20, 25])
    logging.info('Completed')
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')
    parser.add_argument('tests', type=str, help='Directory containing result files.')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('-d', '--depth', action='store_true', help='Whether evaluating in 3D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    args = parser.parse_args()

    print(evaluate(args.groundtruths, args.tests, args.loglevel, args.fmt, args.depth, args.solver))
