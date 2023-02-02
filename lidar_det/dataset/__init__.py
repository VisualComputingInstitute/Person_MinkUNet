from .builder import *
from .utils import *

import numpy as np
from .handles._pypcd import point_cloud_from_fileobj


def load_pcb(file):
    """Load a pcb file.

    Returns:
        pc (np.ndarray[3, N]):
    """
    # pcd_load =
    # o3d.io.read_point_cloud(os.path.join(self.data_dir, url), format='pcd')
    # return np.asarray(pcd_load.points, dtype=np.float32)
    pc = point_cloud_from_fileobj(file).pc_data
    # NOTE: redundent copy, ok for now
    pc = np.array([pc["x"], pc["y"], pc["z"]], dtype=np.float32)
    return pc
