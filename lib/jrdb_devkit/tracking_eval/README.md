# JRDB 2D/3D tracking evaluation script

## Overview

This file describes the JRDB 2D/3D tracking evaluation script.
For overview of the dataset and documentation of the data and label format,
please refer to our website: https://jrdb.stanford.edu

## Label Format

The evaluation script expects a folder in a following structure:

```
cubberly-auditorium-2019-04-22_1/gt/
├── gt.txt
└── 3d_gt.txt
...
tressider-2019-04-26_3/gt/
├── gt.txt
└── 3d_.txt
```

Each subfolder represents a sequence and there is one text file for 2d and 3d 
tracking. The label files contain the following information. All values 
(numerical or strings) are separated via spaces, eachrow corresponds to one 
object. The 13 columns represent:

```
-----------------------------------------------------------------------------
#Values  Name           Description
-----------------------------------------------------------------------------
   1   frame         An integer representing the frame number,
                     where the object appears,
   1   id            An integer identifies a unique ID for the 
                     object, which should be an identical value
                     for the same object in all frames (belonging to
                     a trajectory)
   4   bbox          2D bounding box of an object in the image. 
                     Contains top-left pixel coordinates and
                     box width and height in pixels - <bb_left>,  
                     <bb_top>, <bb_width>, <bb_height>
   3   location      3D object center location in camera coordinates
                     (in meters) i.e. <x>, <y>, <z>,
   3   dimension     3D object dimensions: length, height and width
                     (in meter), i.e. <length>,  <height>, <width>
   1   rotation_y    Rotation ry around Y-axis in camera coordinates 
                     [-pi..pi]	 
   1   conf          Float, indicating confidence in detection
                     (higher is better).
                     * May be an arbitrary value for groundtruth.
-----------------------------------------------------------------------------
```

The conf value contains the detection confidence in the det.txt files. 
For a submission, it acts as a flag whether the entry is to be considered.
A value of 0 means that this particular instance is ignored in the 
evaluation, while any other value can be used to mark it as active. For 
submitted results, all lines in the .txt file with a confidence of 1 are 
considered. Fields which are not used, such as 2D bounding box for 3D 
tracking or location, dimension, and rotation_y for 2D tracking, must be 
set to -1.


## 2D/3D Tracking Benchmark

The primary metric we use to evaluate tracking is MOTA, which combines 
false positives, false negatives, and id switches. We also report MOTP, 
which is a measure of the localisation accuracy of the tracking algorithm.
Rank is determined by MOTA.

## Suggested Validation Split

We provide a suggested validation split to help parameter tune. In the paper,
we show the distribution is very similar to the test dataset. Validation split:

clark-center-2019-02-28_1
gates-ai-lab-2019-02-08_0
huang-2-2019-01-25_0
meyer-green-2019-03-16_0
nvidia-aud-2019-04-18_0
tressider-2019-03-16_1
tressider-2019-04-26_2

## Evaluation Protocol

The MOT conversion script can be used to set up the evaluation groundtruth. 
The website provides details on the file structure under preparing submissions. 
To evaluate tracking results in 2D, run the script:

```
python evaluation/eval_mot.py path/to/sequences path/to/results output_file.txt
``` 
 
As an example:

```
python evaluation/eval_mot.py data/MOT_dataset/sequences/ results/experiment1/ results/experiment1/eval.txt
```

For a detailed explanation of the arguments, run:

```
python evaluation/eval_mot.py -h 
```

The evaluation script expects the groundtruth data to be in the format 
described before. In order to match the results files against the correct 
ground truth files, the results must be in the format:

```
    [sequence_name]_[image_name].txt
    bytes-cafe-2019-02-07_0_image_0.txt
```
 
To evaluate 3d tracking results, just add the --depth flag on the above calls.

```
python evaluation/eval_mot.py MOT/sequences/ results/ MOT_3d_output.txt -d
``` 
 
The results files must be named according to:

```
    [sequence_name]_3d.txt
    bytes-cafe-2019-02-07_0_3d.txt
```


## Acknowledgement

This code is a fork of pymotmetrics:
https://github.com/cheind/py-motmetrics
