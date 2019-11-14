import numpy as np
from laspy.file import File

from redhawkmaster import lasmaster as lm


def rh_add_pid(infile, tile_name, point_id_name="slpid", inc_step=1):
    """
    Add incremental point ID to a tile.

    :param tile_name: name of the output las file
    :param infile: laspy object on which to make a dimension with name point_id_name
    :param point_id_name: name of the dimension
    :param inc_step: how much to increment the point ID.
    :return:
    """

    dimensions = [spec.name for spec in infile.point_format]

    outFile = File(tile_name, mode="w", header=infile.header)

    if not (point_id_name in dimensions):
        outFile.define_new_dimension(name=point_id_name, data_type=7, description='Semantic Landscapes PID')

    outFile.writer.set_dimension(point_id_name, np.arange(len(infile), step=inc_step, dtype=np.uint64))

    for dim in dimensions:
        if dim != point_id_name:
            dat = infile.reader.get_dimension(dim)
            outFile.writer.set_dimension(dim, dat)

    return outFile
