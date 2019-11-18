import numpy as np
from laspy.file import File

from redhawkmaster import lasmaster as lm


def point_id(infile, tile_name, point_id_name="slpid", start_value=0, inc_step=1):
    """
    Add incremental point ID to a tile.

    :param tile_name: name of the output las file
    :param infile: laspy object on which to make a dimension with name point_id_name
    :param point_id_name: name of the dimension
    :param start_value: where the point id dimension will start
    :param inc_step: how much to increment the point ID.
    :return:
    """

    # Get all dimensions that the file have
    dimensions = [spec.name for spec in infile.point_format]

    # Make output las file
    outFile = File(tile_name, mode="w", header=infile.header)

    # Check if we have the point id name in the dimensions
    # if not make the dimension
    if not (point_id_name in dimensions):
        outFile.define_new_dimension(name=point_id_name, data_type=7, description='Semantic Landscapes PID')

    # Set the dimension with the right ids
    outFile.writer.set_dimension(point_id_name, np.arange(start=start_value, stop=(len(infile)*inc_step)+start_value,
                                                          step=inc_step,
                                                          dtype=np.uint64))

    # Populate the points from the input file
    for dim in dimensions:
        if dim != point_id_name:
            dat = infile.reader.get_dimension(dim)
            outFile.writer.set_dimension(dim, dat)

    # Return the las file
    return outFile


def add_hag(tile_name, output_file, vox=1, alpha=0.01):
    """
    Add hag in a file.

    :param output_file: Name of the output file
    :param tile_name: Name of the input file
    :param vox: Voxel size
    :param alpha: A number between 0 and 1 to determine which z value is picked for ground.
    :return:
    """

    cf = {
        "vox": vox,
        "alpha": alpha,
    }

    out_file = lm.lpinteraction.add_hag(tile_name, output_file, config=cf)

    return out_file

