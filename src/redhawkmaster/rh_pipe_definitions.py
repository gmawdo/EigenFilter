from redhawkmaster.rh_inmemory import RedHawkPointCloud
import numpy as np


def point_id(infile: RedHawkPointCloud,
             point_id_name: str,
             start_value: int = 0,
             inc_step: int = 1):
    """
    Add incremental point ID to a tile.
    :param infile: RedHawkPointCloud to have point id added
    :param point_id_name: name of the dimension
    :param start_value: where the point id dimension will start
    :param inc_step: how much to increment the point ID.
    :return:
    """

    pid = np.arange(start=start_value,
                    stop=(len(infile) * inc_step) + start_value,
                    step=inc_step,
                    dtype=np.uint64)

    infile.add_dimension(point_id_name, pid.dtype)

    setattr(infile, point_id_name, pid)

    return None
