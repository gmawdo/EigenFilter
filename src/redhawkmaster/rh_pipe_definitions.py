from redhawkmaster.rh_inmemory import RedHawkPointCloud
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import numpy as np
from pandas import DataFrame

# background functions go here
def uicondition2mask(range):
    """
    The ui sometimes gives the user the option to enter a range. This looks something like
    range = (0,1),(2,4),[-1,-0.5],[10,...], (...,-10)
    This means that the we need to select points with:
    0<x<1 or 2<x<4 or -1<=x<=0.5 or 10<=x or x<-10
    We need to take the range and get a function of x which produces a mask. This is what this function does.
    @param range: The range gotten from the UI
    @return: A function we can apply to x (the vector to be ranged) which outputs a mask
    """
    condition1 = lambda x: (isinstance(x, tuple) or isinstance(x, list))
    condition2 = lambda x: (len(x) == 1) or (len(x) == 2)
    for item in range:
        assert condition1(item) and isinstance(range, list), "ranges should be list of lists, or a list of tuples"
        assert condition2(item), "items in the range should be length 1 or 2"
    f = lambda x: np.zeros(len(x), dtype=bool)

    rh_or = lambda f, g: lambda x: (f(x) | g(x))
    rh_and = lambda f, g: lambda x: (f(x) & g(x))

    gt = lambda t: lambda x: x > t[0]
    lt = lambda t: lambda x: x < t[-1]
    geq = lambda t: lambda x: x >= t[0]
    leq = lambda t: lambda x: x <= t[-1]

    for item in range:
        cond = lambda x: np.ones(len(x), dtype=bool)
        if isinstance(item, list):
            if item[0] is ...:
                pass
            else:
                cond = rh_and(cond, geq(item))
            if item[-1] is ...:
                pass
            else:
                cond = rh_and(cond, leq(item))
        if isinstance(item, tuple):
            if item[0] is ...:
                pass
            else:
                cond = rh_and(cond, gt(item))
            if item[-1] is ...:
                pass
            else:
                cond = rh_and(cond, lt(item))
        f = rh_or(f, cond)

    return f

def clustering(coords, tolerance, min_length, min_pts):
    """
    THIS IS A PURE FUNCTION - NOT FOR USE BY END USER
    :param coords: points to cluster.
    :param tolerance: how close do two points have to be in order to be in same cluster?
    :param max_length: how long can a cluster be?
    :param min_pts: how many points must a cluster have?
    :return:
    """
    if coords.shape[0] >= 1:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        clustering = DBSCAN(eps=tolerance, min_samples=min_pts).fit(coords)
        labels = clustering.labels_
        frame = {
            'A': labels,
            'X': x,
            'Y': y,
            'Z': z,
        }
        df = DataFrame(frame)
        maxs = (df.groupby('A').max()).values
        mins = (df.groupby('A').min()).values
        unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
        lengths = np.sqrt((maxs[inv, 0] - mins[inv, 0]) ** 2 + (maxs[inv, 1] - mins[inv, 1]) ** 2)
        labels[lengths < min_length] = -1
    else:
        labels = np.zeros(coords.shape[0], dtype=int)
    return labels

# pipe definitions go here
def point_id(in_memory: RedHawkPointCloud,
             point_id_name: str,
             start_value: int = 0,
             inc_step: int = 1):
    """
    Add incremental point ID to a tile.
    :param in_memory: RedHawkPointCloud to have point id added
    :param point_id_name: name of the dimension
    :param start_value: where the point id dimension will start
    :param inc_step: how much to increment the point ID.
    :return:
    """

    pid = np.arange(start=start_value,
                    stop=(len(in_memory) * inc_step) + start_value,
                    step=inc_step,
                    dtype=np.uint64)

    in_memory.add_dimension(point_id_name, pid.dtype)

    setattr(in_memory, point_id_name, pid)

    return None


def cluster_labels(in_memory,
                   select_attribute,
                   select_range,
                   distance,
                   min_pts,
                   cluster_attribute,
                   minimum_length):
    """
    Inputs a file and a classification to cluster. Outputs a file with cluster labels.
    Clusters with label 0 are non-core points, i.e. points without "min_pts" within
    "tolerance" (see DBSCAN documentation), or points outside the classification to cluster.
    :param in_memory: RedHawkPointCloud to have cluster labels added
    :param select_attribute: the attribute which you want to use to select a range from
    :param select_range: python list of values to cluster e.g. for classification [3,4,5] for the three types of veg
    :param distance: how close must two points be to be put in the same cluster
    :param min_pts: minimum number of points each point must have in a radius of size "distance"
    :param cluster_attribute: the name given to the clustering labels
    :param minimum_length: the minimum length of a cluster
    """

    x = in_memory.x
    y = in_memory.y
    z = in_memory.z
    # make a vector to store labels
    labels_allpts = np.zeros(len(in_memory), dtype=int)
    # get the point positions
    coords = np.stack((x, y, z), axis=1)
    # make the clusters
    attr = getattr(in_memory, select_attribute)
    mask = uicondition2mask(select_range)(attr)
    labels = 1 + clustering(coords[mask], distance, minimum_length, min_pts)
    # assign the target classification's labels
    labels_allpts[mask] = labels  # find our labels (DBSCAN starts at -1 and we want to start at 0, so add 1)
    # make the output file
    data_types = in_memory.datatypes
    # add new dimension
    if cluster_attribute not in data_types:
        in_memory.add_dimension(cluster_attribute, np.int64)
    # set new dimension to labels
    setattr(in_memory, cluster_attribute, labels_allpts)

    return None
