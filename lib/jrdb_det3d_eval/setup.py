from setuptools import setup, find_packages

# Modified from https://github.com/sshaoshuai/PointRCNN/tree/master/tools/kitti_object_eval_python

# NOTE Only intended as a rough estimate and gives 1-4 higher AP than the official
# C++ evaluation code. Does not work for some sequences, possibily because of bounding
# boxes above ground. Takes about 6 minutes for the validation set on my computer

setup(
    name='jrdb_det3d_eval',
    packages=find_packages(),
)
