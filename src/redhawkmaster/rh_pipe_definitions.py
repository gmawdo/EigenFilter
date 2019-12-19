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


def cluster_labels(infile,
                   attribute,
                   range_to_cluster,
                   distance,
                   min_pts,
                   cluster_attribute,
                   minimum_length):
    """
    Inputs a file and a classification to cluster. Outputs a file with cluster labels.
    Clusters with label 0 are non-core points, i.e. points without "min_pts" within
    "tolerance" (see DBSCAN documentation), or points outside the classification to cluster.
    :param infile: RedHawkPointCloud to have cluster labels added
    :param attribute: the attribute which you want to use to select a range from
    :param range_to_cluster: python list of values to cluster e.g. for classification [3,4,5] for the three types of veg
    :param distance: how close must two points be to be put in the same cluster
    :param min_pts: minimum number of points each point must have in a radius of size "distance"
    :param cluster_attribute: the name given to the clustering labels
    :param minimum_length: the minimum length of a cluster
    """

    x = infile.x
    y = infile.y
    z = infile.z
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions
    coords = np.stack((x, y, z), axis=1)
    # make the clusters
    attr = getattr(infile, attribute)
    mask = uicondition2mask(range_to_cluster)(attr)
    labels = 1 + clustering(coords[mask], distance, minimum_length, min_pts)
    # assign the target classification's labels
    labels_allpts[mask] = labels  # find our labels (DBSCAN starts at -1 and we want to start at 0, so add 1)
    # make the output file
    out_file = File(outfile, mode="w", header=infile.header)
    dimensions = [spec.name for spec in infile.point_format]
    # add new dimension
    if cluster_attribute not in dimensions:
        out_file.define_new_dimension(name=cluster_attribute, data_type=6, description="clustering labels")
    # add pre-existing point records
    for dimension in dimensions:
        dat = infile.reader.get_dimension(dimension)
        out_file.writer.set_dimension(dimension, dat)
    # set new dimension to labels
    out_file.writer.set_dimension(cluster_attribute, labels_allpts)
    out_file.close()

    return None
