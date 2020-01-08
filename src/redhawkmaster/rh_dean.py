import numpy as np
from laspy.file import File
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import lasmaster as lm
from redhawkmaster.las_modules import virus_background
import pandas as pd
import os


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
    outFile.writer.set_dimension(point_id_name,
                                 np.arange(start=start_value, stop=(len(infile) * inc_step) + start_value,
                                           step=inc_step,
                                           dtype=np.uint64))

    # Populate the points from the input file
    for dim in dimensions:
        if dim != point_id_name:
            dat = infile.reader.get_dimension(dim)
            outFile.writer.set_dimension(dim, dat)

    # Return the las file
    return outFile


def bb(x, y, z, predicate, R):
    Coords = np.stack((x, y), axis=0)
    coords_R = np.matmul(R, Coords[:, predicate])
    x_max = np.amax(coords_R[:, 0, :], axis=-1)
    y_max = np.amax(coords_R[:, 1, :], axis=-1)
    x_min = np.amin(coords_R[:, 0, :], axis=-1)
    y_min = np.amin(coords_R[:, 1, :], axis=-1)
    A = (x_max - x_min) * (y_max - y_min)
    k = np.argmin(A)
    R_min = R[k, :, :]
    Coords_R = np.matmul(R[k, :, :], Coords)
    predicate_bb = (x_min[k] - 0.25 <= Coords_R[0]) & (y_min[k] - 0.25 <= Coords_R[1]) & (
            x_max[k] >= Coords_R[0] - 0.25) & (y_max[k] >= Coords_R[1] - 0.25) & (
                           max(z[predicate]) + 0.5 >= z) & (min(z[predicate]) - 0.5 <= z)
    return predicate_bb, A[k], min(x[predicate_bb]), max(x[predicate_bb]), min(y[predicate_bb]), max(
        y[predicate_bb])


def bbox(tile_name, output_file):
    inFile = File(tile_name)
    hag = inFile.hag
    # num == accuracy
    theta = np.linspace(0, 2 * np.pi, num=1000)
    S = np.sin(theta)
    C = np.cos(theta)
    R = np.zeros((theta.size, 2, 2))
    R[:, 0, 0] = np.cos(theta)
    R[:, 0, 1] = -np.sin(theta)
    R[:, 1, 0] = np.sin(theta)
    R[:, 1, 1] = np.cos(theta)

    x = inFile.x
    y = inFile.y
    z = inFile.z
    coords = np.stack((x, y), axis=1)
    out = File(output_file, mode="w", header=inFile.header)
    out.points = inFile.points
    classn = np.ones(len(inFile), dtype=int)
    classn[:] = inFile.classification[:]

    # ===== END ===== Bounding Box - Rectangle

    # ===== START ===== Corridor 2D
    if (classn == 0).any() and (classn == 1).any():
        nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(np.stack((x, y), axis=1)[classn == 1, :])
        distances, indices = nhbrs.kneighbors(np.stack((x, y), axis=1)[classn == 0, :])
        classn0 = classn[classn == 0]

        try:
            angle = inFile.ang3
        except AttributeError:
            angle = inFile.ang2

        # Distance Threshold ==> 1  && Angle 0.2
        classn0[(distances[:, 0] < 1) & (angle[classn == 0] < 0.2)] = 5
        classn0[(distances[:, 0] < 1) & (angle[classn == 0] < 0.2)] = 5
        classn[classn == 0] = classn0
        # ===== END ===== Corridor 2D

        # ===== START ===== Voxel 2D Analysis
        if (classn == 5).any():
            classn_5_save = classn == 5
            classn5 = classn[classn_5_save]
            unq, ind, inv = np.unique(np.floor(np.stack((x, y), axis=1)[classn_5_save, :]).astype(int),
                                      return_index=True, return_inverse=True, return_counts=False, axis=0)
            for item in np.unique(inv):
                z_max = np.max(z[classn_5_save][inv == item])
                z_min = np.min(z[classn_5_save][inv == item])
                # Range attribute 5
                if (z_max - z_min) < 5:
                    classn5[inv == item] = 0
            classn[classn_5_save] = classn5
        # ===== END ===== Voxel 2D Analysis
        # ===== START ===== Small tool
        if (classn == 5).any():
            classn_5_save = classn == 5
            clustering = DBSCAN(eps=0.5, min_samples=1).fit(np.stack((x, y), axis=1)[classn_5_save, :])
            labels = clustering.labels_
            L = np.unique(labels)

            for item in L:
                predicate = np.zeros(len(inFile), dtype=bool)[(classn == 0) | classn_5_save]
                predicate[classn_5_save[(classn == 0) | classn_5_save]] = labels == item
                predicate_bb, area, x_min, x_max, y_min, y_max = bb(x[(classn == 0) | classn_5_save],
                                                                    y[(classn == 0) | classn_5_save],
                                                                    z[(classn == 0) | classn_5_save], predicate, R)
                classn05 = classn[(classn == 0) | classn_5_save]
                classn05[predicate_bb] = 5
                classn[(classn == 0) | classn_5_save] = classn05
        # ===== END ===== Small tool
    out.classification = classn
    out.close()


def bbox_rectangle(inFile, out, classification_in=2, accuracy=3601):
    """
    Adds minimal area rectangle around some classification.

    :param classification_in:
    :param inFile:
    :param out:
    :param accuracy:
    :return:
    """

    theta = np.linspace(0, 2 * np.pi, num=accuracy)

    R = np.zeros((theta.size, 2, 2))
    R[:, 0, 0] = np.cos(theta)
    R[:, 0, 1] = -np.sin(theta)
    R[:, 1, 0] = np.sin(theta)
    R[:, 1, 1] = np.cos(theta)

    x = inFile.x
    y = inFile.y
    z = inFile.z

    classn = np.ones(len(inFile), dtype=int)
    classn[:] = inFile.classification[:]

    classn_2_save = classn == classification_in

    if (classn == classification_in).any():
        clustering = DBSCAN(eps=0.5, min_samples=1).fit(np.stack((x, y, z), axis=1)[classn_2_save, :])
        labels = clustering.labels_
        L = np.unique(labels)
        bldgs = np.empty((L.size, 6))
        i = 0
        for item in L:
            predicate = np.zeros(len(inFile), dtype=bool)
            predicate[classn_2_save] = labels == item
            predicate_bb, area, x_min, x_max, y_min, y_max = bb(x, y, z, predicate, R)
            classn[predicate_bb] = classification_in
            bldgs[i] = [i, area, x_min, x_max, y_min, y_max]
            i += 1
            np.savetxt("buildings_" + inFile.filename[-4:] + ".csv", bldgs, delimiter=",",
                       header="ID, Area, X_min, X_max, Y_min, Y_max")

    out.classification = classn

    return classn


def corridor_2d(inFile, distance_threshold=1, angle_threshold=0.2,
                classification_cond=1, classification_pyl=5, classification_up=0):
    """
    Take all unclassified points within xy distance 1m of the
    conductor . Set all such points with vertical
    angle < 0.2 to classification 5.

    :param classification_up:
    :param classification_pyl: Classification for a pylon.
    :param classification_cond: Classification for a conductor.
    :param inFile: laspy file with write mode
    :param distance_threshold: distance threshold
    :param angle_threshold: angle threshold
    :return:
    """
    classn = inFile.classification
    x = inFile.x
    y = inFile.y

    if (classn == classification_up).any() and (classn == classification_cond).any():
        nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(np.stack((x, y), axis=1)
                                                                         [classn == classification_cond, :])
        distances, indices = nhbrs.kneighbors(np.stack((x, y), axis=1)[classn == classification_up, :])
        classn0 = classn[classn == classification_up]

        try:
            angle = inFile.ang3
        except AttributeError:
            angle = inFile.ang2

        classn0[(distances[:, 0] < distance_threshold) & (angle[classn == classification_up] < angle_threshold)] = \
            classification_pyl
        classn[classn == classification_up] = classn0

    inFile.classification = classn
    inFile.classification = classn

    return classn


def voxel_2d(infile, height_threshold=5, classification_in=5, classification_up=0):
    """
    Set up 1mx1m voxels and within each, select class 5
    points and set their class to 0 if the range (= max - min)
    of z values of class 5 points is < 5. This is a slightly ad
    hoc addition to get over the leaky affect of bounding
    boxes.

    :param infile: las file with output mode
    :param height_threshold: range attribute
    :param classification_in: classification of the pylon
    :param classification_up: classification of what is unclassified
    :return:
    """
    x = infile.x
    y = infile.y
    z = infile.z
    classn = infile.classification

    if (classn == classification_in).any():
        classn_5_save = classn == classification_in
        classn5 = classn[classn_5_save]
        unq, ind, inv = np.unique(np.floor(np.stack((x, y), axis=1)[classn_5_save, :]).astype(int),
                                  return_index=True, return_inverse=True, return_counts=False, axis=0)
        for item in np.unique(inv):
            z_max = np.max(z[classn_5_save][inv == item])
            z_min = np.min(z[classn_5_save][inv == item])
            # Range attribute 5
            if (z_max - z_min) < height_threshold:
                classn5[inv == item] = classification_up
        classn[classn_5_save] = classn5

    infile.classification = classn

    return classn


def recover_un(infile, classification_up=0, classification_in=5, accuracy=1000):
    """
    Cluster class 2 points. Use bounding box to recover
    some unclassified points and make them into class 2, if
    they fall within the bounding box.

    :param accuracy: how large will be the accuracy
    :param infile: laspy file on which to change the classification
    :param classification_up: on which classification to update
    :param classification_in: classification that you want to be recovered in classification up
    :return: array of the classification
    """
    theta = np.linspace(0, 2 * np.pi, num=accuracy)

    R = np.zeros((theta.size, 2, 2))
    R[:, 0, 0] = np.cos(theta)
    R[:, 0, 1] = -np.sin(theta)
    R[:, 1, 0] = np.sin(theta)
    R[:, 1, 1] = np.cos(theta)
    classn = infile.classification
    x = infile.x
    y = infile.y
    z = infile.z
    if (classn == classification_in).any():
        classn_5_save = classn == classification_in
        clustering = DBSCAN(eps=0.5, min_samples=1).fit(np.stack((x, y), axis=1)[classn_5_save, :])
        labels = clustering.labels_
        L = np.unique(labels)

        for item in L:
            predicate = np.zeros(len(infile), dtype=bool)[(classn == classification_up) | classn_5_save]
            predicate[classn_5_save[(classn == classification_up) | classn_5_save]] = labels == item
            predicate_bb, area, x_min, x_max, y_min, y_max = bb(x[(classn == classification_up) | classn_5_save],
                                                                y[(classn == classification_up) | classn_5_save],
                                                                z[(classn == classification_up) | classn_5_save],
                                                                predicate, R)
            classn05 = classn[(classn == classification_up) | classn_5_save]
            classn05[predicate_bb] = classification_in
            classn[(classn == classification_up) | classn_5_save] = classn05
    infile.classification = classn

    return classn


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


def add_attributes(tile_name, output_file, time_intervals=10, k=range(4, 50), radius=0.5,
                   virtual_speed=0, voxel_size=0):
    """
    Add some math magic to a tile.

    :param tile_name: name of the input file
    :param output_file: name of the output_file
    :param radius: Furtherst distance allowed for neighbours (i.e. radius of sphere around points)
    :param k: list of k values to consider
    :param time_intervals: how many time intervals to split the tile into for
    computing attributes (this only affects memory usage - no edge effects)
    :param virtual_speed: Most unintuitive parameter. This is how much time is considered as a spacial variable along
     with x,y,z in spacetime solution. Wehave used 2.0 before. 0 is no spacetime.
    :param voxel_size:
    :return:
    """
    # Prepare the config
    cf = {
        "timeIntervals": time_intervals,
        "k": k,  # must be a generator
        "radius": radius,
        "virtualSpeed": virtual_speed,
        "decimate": voxel_size,
    }

    # Call the lasmaster
    lm.lpinteraction.attr(tile_name, output_file, config=cf)


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


def corridor(coords, eigenvectors, mask, R, S):
    """
    Find a cylindrical corridor around a family of points.

    This function takes a family of points, encoded in a numpy array coords; coords.shape == (n, d) where
    n is the number of points and d is the number of dimensions. Note that eigenvectors.shape == coords[mask,:].shape
    :param coords: all the points you want to drive the corridor through
    :param eigenvectors: the eigenvectors of points you want to drive the corridor around
    :param mask: the mask describing the points you want to drive the corridor around
    :param R: the radius of the corridor
    :param S: the extension of the corridor
    :return: the mask of the points in the corridor
    """
    # apply the mask to the coordinates to get the points we are interested in drawing a corridor around
    # The next uncommented line would be needed to match the original demo, but it shouldn't really have been used
    # in the first place!
    # coords = 0.05*np.floor(coords/0.05)
    if mask.any():
        coord_mask = coords[mask, :]
        # find nearest neighbours from each point to the points of interest
        nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coord_mask)
        distances, indices = nhbrs.kneighbors(coords)
        nearest_nbrs = indices[:, 0]

        # find direction from each point to point of interest, project onto eigenvector
        v = eigenvectors[nearest_nbrs]
        u = coords - coords[mask][nearest_nbrs]
        scale = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1] + u[:, 2] * v[:, 2]  # np.sum(u * v, axis=1)
        # find coprojection
        w = u - scale[:, None] * v

        # find distance to line formed by eigenvector
        w_norms = np.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2 + w[:, 2] ** 2)  # np.sqrt(np.sum(w ** 2, axis=1))

        # return condition for the corridor
        condition = (w_norms < R) & (np.absolute(scale) < S)
    else:
        condition = np.zeros(coords.shape[-2], dtype=bool)

    return condition


def eigenvector_corridors_v01_0(infile,
                                outfile,
                                attribute_to_corridor,
                                range_to_corridor,
                                protection_attribute,
                                range_to_protect,
                                classification_of_corridor,
                                radius_of_cylinders,
                                length_of_cylinders):
    """
    Select some points to put a corridor (=collection of cylinders) around, by using an attribute and range.
    Select an attribute range to protect from change in classification.
    Choose classification for points inside corridor, with radius and length of cylinders to define corridor.
    @param infile: input file name
    @param outfile: output file name
    @param attribute_to_corridor: attribute used to select points
    @param range_to_corridor: range of values to change
    @param protection_attribute: attribute used to protect points from classification change
    @param range_to_protect: range of values to protect
    @param classification_of_corridor: classification of inside corridor
    @param radius_of_cylinders: radius of cylinders which describe corridor
    @param length_of_cylinders: length of cylinders which describe corridor
    @return: file with selection reclassified
    """
    inFile = File(infile)
    classn = 1 * inFile.classification
    outFile = File(outfile, mode="w", header=inFile.header)
    outFile.points = inFile.points
    mask = uicondition2mask(range_to_corridor)(inFile.reader.get_dimension(attribute_to_corridor))
    protect = uicondition2mask(range_to_protect)(inFile.reader.get_dimension(protection_attribute))
    coords = np.stack((inFile.x, inFile.y, inFile.z), axis=1)
    eigenvectors = np.stack((inFile.eig20, inFile.eig21, inFile.eig22), axis=1)[mask, :]
    condition = corridor(coords, eigenvectors, mask, radius_of_cylinders,
                         length_of_cylinders)
    classn[condition & (~ protect)] = classification_of_corridor
    outFile.classification = classn
    outFile.close()


def eigenvector_corridors_v01_1(infile,
                                outfile,
                                attribute_to_corridor,
                                range_to_corridor,
                                protection_attribute,
                                range_to_protect,
                                classification_of_corridor,
                                radius_of_cylinders,
                                length_of_cylinders):
    """
    Select some points to put a corridor (=collection of cylinders) around, by using an attribute and range.
    Select an attribute range to protect from change in classification.
    Choose classification for points inside corridor, with radius and length of cylinders to define corridor.
    :param infile: input file name
    :param outfile: output file name
    :param attribute_to_corridor: attribute used to select points
    :param range_to_corridor: range of values to change
    :param protection_attribute: attribute used to protect points from classification change
    :param range_to_protect: range of values to protect
    :param classification_of_corridor: classification of inside corridor
    :param radius_of_cylinders: radius of cylinders which describe corridor
    :param length_of_cylinders: length of cylinders which describe corridor
    :return: file with selection reclassified
    """

    classn = 1 * infile.classification

    mask = uicondition2mask(range_to_corridor)(infile.reader.get_dimension(attribute_to_corridor))
    protect = uicondition2mask(range_to_protect)(infile.reader.get_dimension(protection_attribute))
    coords = np.stack((infile.x, infile.y, infile.z), axis=1)
    eigenvectors = np.stack((infile.eig20, infile.eig21, infile.eig22), axis=1)[mask, :]
    condition = corridor(coords, eigenvectors, mask, radius_of_cylinders,
                         length_of_cylinders)
    classn[condition & (~ protect)] = classification_of_corridor
    outfile.classification = classn


def point_dimension(inFile):
    """
    Dimensions for each point of a las file.

    :param inFile: las file with dim1, dim2 and dim3 attributes.
    :return: Dimensions for each point
    """
    # gather the three important attributes into an array
    lps = np.stack((inFile.dim1, inFile.dim2, inFile.dim3), axis=1)
    # compute dimensions
    dims = 1 + np.argmax(lps, axis=1)

    return dims


def dimension1d2d3d_v01_0(infile,
                          outfile):
    """
    Adds dimension 1, 2 or 3 to each point.
    :param infile: name of infile
    :param outfile: name of outfile
    :return: nothing, just writes new file
    """
    in_file = File(infile)
    out_file = File(outfile, mode="w", header=in_file.header)
    # add dimension
    dimensions = [spec.name for spec in in_file.point_format if spec.name != "dim"]
    out_file.define_new_dimension(name="dimension1d2d3d", data_type=6, description="dimension")
    # add pre-existing point records
    for dimension in dimensions:
        dat = in_file.reader.get_dimension(dimension)
        out_file.writer.set_dimension(dimension, dat)
    # add new dimension
    out_file.writer.set_dimension("dimension1d2d3d", point_dimension(infile))
    out_file.close()


def dimension1d2d3d_v01_1(infile,
                          outfile):
    """
    Adds dimension 1, 2 or 3 to each point.
    :param infile: name of infile
    :param outfile: name of outfile
    :return: nothing, just writes new file
    """

    outfile.writer.set_dimension("dimension1d2d3d", point_dimension(infile))


def eigen_clustering(coords, eigenvector, tolerance, eigenvector_scale, max_length, min_pts):
    """
    THIS IS A PURE FUNCTION - NOT FOR USE BY END USER
    ADD DESCRIPTION Eigenvectors should have unit length.
    :param coords: points_to_cluster.
    :param eigenvector: eigenvectors for each point
    :param tolerance: tolerance - how close do two points have to be in order to be in same cluster?
    :param eigenvector_scale:
    :param max_length: how long can a cluster be?
    :param min_pts: how many points must a cluster have?
    :return: labels
    """

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    v0 = eigenvector_scale * eigenvector[:, 0]
    v1 = eigenvector_scale * eigenvector[:, 1]
    v2 = eigenvector_scale * eigenvector[:, 2]
    condition = ((v0 >= 0) & (v1 < 0) & (v2 < 0)) | ((v1 >= 0) & (v2 < 0) & (v0 < 0)) | (
            (v2 >= 0) & (v0 < 0) & (v1 < 0)) | ((v0 < 0) & (v1 < 0) & (v2 < 0))
    v0[condition] = -v0[condition]
    v1[condition] = -v1[condition]
    v2[condition] = -v2[condition]
    clusterable = np.stack((v0, v1, v2, x, y, z), axis=1)
    clustering = DBSCAN(eps=tolerance, min_samples=min_pts).fit(clusterable)
    labels = clustering.labels_
    frame = {
        'A': labels,
        'X': x,
        'Y': y,
        'Z': z,
    }
    df = pd.DataFrame(frame)
    maxs = (df.groupby('A').max()).values
    mins = (df.groupby('A').min()).values
    unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
    lengths = np.sqrt((maxs[inv, 0] - mins[inv, 0]) ** 2 + (maxs[inv, 1] - mins[inv, 1]) ** 2)
    labels[lengths < max_length] = -1
    return labels


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
        df = pd.DataFrame(frame)
        maxs = (df.groupby('A').max()).values
        mins = (df.groupby('A').min()).values
        unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
        lengths = np.sqrt((maxs[inv, 0] - mins[inv, 0]) ** 2 + (maxs[inv, 1] - mins[inv, 1]) ** 2)
        labels[lengths < min_length] = -1
    else:
        labels = np.zeros(coords.shape[0], dtype=int)
    return labels


def add_classification(input_file, output_file):
    """
    :param filename:
    :return:
    """
    # load the voxel numbers - they will be only one (0) if no voxelisation happened
    inFile = File(input_file)
    # find how to map each point onto each voxel
    voxel = inFile.vox
    # find how to map each point onto each voxel
    UNQ, IND, INV, CNT = np.unique(voxel, return_index=True, return_inverse=True, return_counts=True)
    # determine by number of voxel numbers whether decimation occured
    decimated = IND.size > 1
    # if no decimation occured this mapping must be trivial:
    if not decimated:
        IND = np.arange(len(inFile))
        INV = IND

    # grab the attributes we need - but only on decimated points
    x = inFile.x[IND]
    y = inFile.y[IND]
    z = inFile.z[IND]
    eig2 = inFile.eig2[IND]

    # scale down coordinates
    if decimated:
        u = inFile.dec
        coords = u[:, None] * np.floor(np.stack((x / u, y / u, z / u), axis=1))
    else:
        coords = np.stack((x, y, z), axis=1)
    # build the probabilistic dimension

    dims = point_dimension(inFile)[IND]

    classn = 1 * inFile.classification[IND]
    classn[:] = 0
    noise = eig2 < 0
    dim1 = dims == 1
    dim2 = dims == 2
    dim3 = dims == 3

    mask = dim1
    if mask.any():
        v0 = 1 * inFile.eig20[IND]
        v1 = 1 * inFile.eig21[IND]
        v2 = 1 * inFile.eig22[IND]
        line_of_best_fit_direction = np.stack((v0, v1, v2), axis=1)
        labels = eigen_clustering(coords[mask], line_of_best_fit_direction[mask], 0.5, 5, 2, 1)
        class_mask = classn[mask]
        class_mask[:] = 1
        class_mask[labels == -1] = 0
        classn[mask] = class_mask

        conductor = corridor(coords, line_of_best_fit_direction[classn == 1], classn == 1, R=0.5, S=2)
        classn[conductor] = 1
        classn[noise] = 7

    mask = dim2 & (~ noise) & (classn != 1)
    if mask.any():
        v0 = 1 * inFile.eig10[IND]
        v1 = 1 * inFile.eig11[IND]
        v2 = 1 * inFile.eig12[IND]
        plane_of_best_fit_direction = np.stack((v0, v1, v2),
                                               axis=1)  # np.sqrt(v0 ** 2 + v1 ** 2 + v2 ** 2)[:, None]
        labels = eigen_clustering(coords[mask], plane_of_best_fit_direction[mask], 0.5, 5, 2, 1)
        class_mask = classn[mask]
        class_mask[:] = 2
        class_mask[labels == -1] = 0
        classn[mask] = class_mask
        if (classn == 2).any():
            nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coords[classn == 2])
            distances, indices = nhbrs.kneighbors(coords)
            classn[(distances[:, 0] < 0.5) & (classn != 7) & (classn != 1)] = 2

    mask = dim3 & (~ noise) & (classn != 1) & (classn != 2)
    if mask.any():
        labels = clustering(coords[mask], 0.5, 2, 1)
        class_mask = classn[mask]
        class_mask[:] = 3
        class_mask[labels == -1] = 0
        classn[mask] = class_mask

    if (classn == 3).any():
        nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coords[classn == 3])
        distances, indices = nhbrs.kneighbors(coords)
        classn[(distances[:, 0] < 0.5) & (classn != 7) & (classn != 1) & (classn != 2)] = 3

    if ((classn != 0) & (classn != 7)).any():
        nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coords[(classn != 0) & (classn != 7), :])
        distances, indices = nhbrs.kneighbors(coords[classn == 0, :])
        classn0 = classn[classn == 0]
        classn0[(distances[:, 0] < 0.5)] = (classn[(classn != 0) & (classn != 7)])[
            indices[(distances[:, 0] < 0.5), 0]]
        classn[(classn == 0)] = classn0

    outFile = File(output_file, mode="w", header=inFile.header)
    outFile.points = inFile.points
    outFile.classification = classn[INV]
    outFile.close()


def conductor_matters_1(infile, epsilon=2.5, classification_in=0, classification_up=1,
                        distance_ground=7, length_threshold=4):
    """

    :param infile:
    :param epsilon:
    :param classification_up:
    :param classification_in:
    :param length_threshold:
    :param distance_ground:
    :return:
    """

    x = infile.x
    y = infile.y
    z = infile.z
    hag = infile.hag
    classn = infile.classification
    cond = classn == classification_up
    if cond.any():
        clustering = DBSCAN(eps=epsilon, min_samples=1).fit(np.stack((x, y, z), axis=1)[cond, :])
        labels = clustering.labels_
        frame = {
            'A': labels,
            'X': x[cond],
            'Y': y[cond],
            'Z': z[cond],
            'H': hag[cond]
        }
        df = pd.DataFrame(frame)
        maxs = (df.groupby('A').max()).values
        mins = (df.groupby('A').min()).values
        lq = (df.groupby('A').quantile(0.5)).values
        unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
        lengths = \
            np.sqrt((maxs[:, 0] - mins[:, 0]) ** 2 + (maxs[:, 1] - mins[:, 1]) ** 2 + (maxs[:, 2] - mins[:, 2]) ** 2)[
                inv]
        hags = lq[inv, 3]
        lengths[labels == -1] = 0
        classn1 = classn[cond]
        classn1[:] = classification_up
        classn1[lengths <= length_threshold] = classification_in
        classn1[hags < distance_ground] = classification_in
        classn[cond] = classn1

    infile.classification = classn

    return classn


def veg_risk(infile, classification_in=1, classification_veg=3, classification_inter=4, distance_veg=3):
    """

    :param infile:ok
    :param classification_in:
    :param classification_veg:
    :param classification_inter:
    :param distance_veg:
    :return:
    """
    classn = infile.classification
    x = infile.x
    y = infile.y
    z = infile.z

    if (classn == classification_in).any() and (classn == classification_veg).any():
        nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(np.stack((x, y, z), axis=1)
                                                                         [classn == classification_in, :])
        distances, indices = nhbrs.kneighbors(np.stack((x, y, z), axis=1)[classn == classification_veg, :])
        classn3 = classn[classn == classification_veg]
        classn3[distances[:, 0] < distance_veg] = classification_inter
        classn[classn == classification_veg] = classn3

    infile.classification = classn


def sd_merge(input_files, output_file):
    """
    Merge array of file into one las file.

    :param input_files: Array of las files ['infile1.las', 'infile2.las']
    :param output_file: Name of the output file
    :return:
    """

    tile1 = input_files[0]
    tile2 = input_files[1]
    name = output_file
    inFile1 = File(tile1)
    inFile2 = File(tile2)
    outFile = File(name, mode="w", header=inFile1.header)
    outFile.x = np.concatenate((inFile1.x, inFile2.x))
    outFile.y = np.concatenate((inFile1.y, inFile2.y))
    outFile.z = np.concatenate((inFile1.z, inFile2.z))
    outFile.gps_time = np.concatenate((inFile1.gps_time, inFile2.gps_time))
    outFile.classification = np.concatenate((inFile1.classification, inFile2.classification))
    outFile.intensity = np.concatenate((inFile1.intensity, inFile2.intensity))
    outFile.return_num = np.concatenate((inFile1.return_num, inFile2.return_num))
    outFile.num_returns = np.concatenate((inFile1.num_returns, inFile2.num_returns))
    mask = (np.concatenate((np.ones(len(inFile1), dtype=int), np.zeros(len(inFile2), dtype=int)))).astype(bool)
    specs1 = [spec.name for spec in inFile1.point_format]
    specs2 = [spec.name for spec in inFile2.point_format]
    for dim in specs1:
        dat1 = inFile1.reader.get_dimension(dim)
        DAT = np.zeros(len(inFile1) + len(inFile2), dtype=dat1.dtype)
        DAT[mask] = dat1
        if dim in specs2:
            dat2 = inFile2.reader.get_dimension(dim)
            DAT[~mask] = dat2
        outFile.writer.set_dimension(dim, DAT)
    outFile.close()


def finish_tile(pair, output_file):
    tile, original = pair[0], pair[1]
    ogFile = File(original)
    inFile = File(tile)
    hag = inFile.hag
    hd = ogFile.header

    outFile = File(output_file, mode="w", header=hd)

    args = np.argsort(inFile.slpid)

    classn = 1 * inFile.classification
    classn0 = classn == 0
    classn1 = classn == 1
    classn2 = classn == 2
    classn3 = classn == 3
    classn4 = classn == 4
    classn5 = classn == 5
    classn6 = classn == 6
    classn[classn0] = 0
    classn[classn1] = 14
    classn[classn2] = 6
    classn[classn3] = 5
    classn[classn4] = 5
    classn[classn5] = 15
    classn[classn6] = 2
    lo = (0.5 < hag) & (hag <= 2)
    med = (2 < hag) & (hag <= 5)
    hi = (5 < hag)
    veg = classn3
    classn[veg] = 0
    classn[lo & veg] = 3
    classn[med & veg] = 4
    classn[hi & veg] = 5
    classn[(classn == 15) & (hag < 2)] = 0
    classn[classn == 7] = 0
    # classn[(hag < 0.5) & (classn != 6)] = 2
    classn[(classn == 14) & (hag < 2)] = 0
    classn[(hag < 0.5) & (classn == 0)] = 2

    for spec in ogFile.point_format:
        outFile.writer.set_dimension(spec.name, inFile.reader.get_dimension(spec.name)[args])
    outFile.classification = classn[args]
    outFile.x = inFile.x[args]
    outFile.y = inFile.y[args]
    outFile.z = inFile.z[args]


def pdal_smrf_enel(input_file, output_file):
    #	ground_command = "pdal ground --initial_distance 1.0 --writers.las.extra_dims=all -i {} -o {}"
    ground_command = "pdal translate " \
                     "--readers.las.extra_dims=\"slpid=uint64\" " \
                     "--writers.las.extra_dims=all {} {} smrf" \
        #	" --filters.smrf.slope={} " \
    #	"--filters.smrf.cut={} " \
    #	"--filters.smrf.window={} " \
    #	"--filters.smrf.cell={} " \
    #	"--filters.smrf.scalar={} " \
    "--filters.smrf.threshold=1.0"
    command = ground_command.format(input_file, output_file)
    # command = ground_command.format(tile, "ground_"+tile)
    os.system(command)
    inFile = File(output_file, mode="rw")
    ground = inFile.classification == 2
    classn = 1 * inFile.classification
    classn[ground] = 6
    classn[~ground] = 0
    inFile.classification = classn
    inFile.close()


def delaunay_triangulation_v01_0(tile,
                                 output_file,
                                 classifications_to_search,
                                 classification_out,
                                 cluster_attribute,
                                 output_ply):
    # triangulates clusters as dictated by cluster_attribute
    # produces a ply file for each tile
    """
    :param tile: input tile name
    :param output_file: output tile name
    :param classifications_to_search: classification in which to search for interior
    :param classification_out: classification of interior
    :param cluster_attribute: name of attribute holding clustering labels
    :param output_ply: boolean (True/False) to dictate whether or not to write a file
    :return:
    """
    from plyfile import PlyData, PlyElement
    inFile = File(tile, mode="r")
    x = inFile.x
    y = inFile.y
    z = inFile.z
    classn = inFile.classification
    # get the cluster labels for the clusters we want to triangulate
    labels = inFile.reader.get_dimension(cluster_attribute)
    # get point positions
    coords = np.stack((x, y, z), axis=1)
    # find the points which come from actual clusters - I called this condition "tree" because it was our first
    # feel free to change
    tree = labels > 0

    # write the ply file
    if output_ply:
        v = 0
        f = 0
        t = 0
        ply_body_v = ""
        ply_body_f = ""
        if tree.any():
            for i in np.unique(labels[tree]):
                condition = labels == i
                if condition[condition].size >= 5:
                    tri = Delaunay(coords[condition, :])
                    simp = tri.find_simplex(coords)
                    include = np.zeros(len(inFile), dtype=bool)
                    for k in classifications_to_search:
                        include[classn == k] = True
                    classn[(simp != -1) & include] = 3
                    vertices = tri.simplices
                    for row in coords[condition, :]:
                        ply_body_v += f"\n{row[0]} {row[1]} {row[2]}"
                        v += 1
                    for row in vertices:
                        ply_body_f += f"\n{4} {row[0] + t} {row[1] + t} {row[2] + t} {row[3] + t}"
                        f += 1
                t += condition[condition].size
        ply_header = "ply\n"
        ply_header += "format ascii 1.0\n"
        ply_header += f"element vertex {v}\n"
        ply_header += "property float32 x\n"
        ply_header += "property float32 y\n"
        ply_header += "property float32 z\n"
        ply_header += f"element face {f}\n"
        ply_header += "property list uint8 int32 vertex_indices\n"
        ply_header += "end_header"
        plyobject = open(output_file[:-4] + '.ply', mode='w')
        plyobject.write(ply_header + ply_body_v + ply_body_f)
        plyobject.close()

    # write the new file with points inside the triangles reclassified
    outFile = File(output_file, mode="w", header=inFile.header)
    outFile.points = inFile.points
    classn[tree] = classification_out
    outFile.classification = classn
    outFile.close()


def cluster_labels_v01_0(infile,
                         outfile,
                         classification_to_cluster,
                         tolerance,
                         min_pts,
                         cluster_attribute,
                         minimum_length):
    """
    Inputs a file and a classification to cluster. Outputs a file with cluster labels.
    Clusters with label 0 are non-core points, i.e. points without "min_pts" within
    "tolerance" (see DBSCAN documentation), or points outside the classification to cluster.
    :param infile: input file name
    :param outfile: output file name
    :param classification_to_cluster: which points do we want to cluster
    :param tolerance: see min_pts
    :param min_pts: minimum number of points each point must have in a radius of size "tolerance"
    :param cluster_attribute: the name given to the clustering labels
    :return:
    """
    # we shouldn't use las_modules.cluster function because it acts on a file, not on a family of points
    infile = File(infile, mode="r")
    x = infile.x
    y = infile.y
    z = infile.z
    classn = infile.classification
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions
    coords = np.stack((x, y, z), axis=1)

    # find our labels (DBSCAN starts at -1 and we want to start at 0, so add 1)
    labels = 1 + clustering(coords[classn == classification_to_cluster], tolerance, minimum_length, min_pts)
    # assign the target classification's labels
    labels_allpts[classn == classification_to_cluster] = labels
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


def cluster_labels_v01_1(infile,
                         outfile,
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
    :param infile: input file name
    :param outfile: output file name
    :param attribute: the attribute which you want to use to select a range from
    :param range_to_cluster: python list of values to cluster e.g. for classification [3,4,5] for the three types of veg
    :param distance: how close must two points be to be put in the same cluster
    :param min_pts: minimum number of points each point must have in a radius of size "distance"
    :param cluster_attribute: the name given to the clustering labels
    :param minimum_length: the minimum length of a cluster
    """
    # we shouldn't use las_modules.cluster function because it acts on a file, not on a family of points
    infile = File(infile, mode="r")
    x = infile.x
    y = infile.y
    z = infile.z
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions
    coords = np.stack((x, y, z), axis=1)
    # make the clusters
    mask = np.zeros(len(infile), dtype=bool)
    for item in range_to_cluster:
        mask[(infile.reader.get_dimension(attribute) >= item[0]) & (
                infile.reader.get_dimension(attribute) <= item[-1])] = True
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


def cluster_labels_v01_2(infile,
                         outfile,
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
    :param infile: input file name
    :param outfile: output file name
    :param attribute: the attribute which you want to use to select a range from
    :param range_to_cluster: python list of values to cluster e.g. for classification [3,4,5] for the three types of veg
    :param distance: how close must two points be to be put in the same cluster
    :param min_pts: minimum number of points each point must have in a radius of size "distance"
    :param cluster_attribute: the name given to the clustering labels
    :param minimum_length: the minimum length of a cluster
    """
    # we shouldn't use las_modules.cluster function because it acts on a file, not on a family of points
    infile = File(infile, mode="r")
    x = infile.x
    y = infile.y
    z = infile.z
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions
    coords = np.stack((x, y, z), axis=1)
    # make the clusters
    attr = infile.reader.get_dimension(attribute)
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


def cluster_labels_v01_3(infile,
                         outfile,
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
    :param infile: input file name
    :param outfile: output file name
    :param attribute: the attribute which you want to use to select a range from
    :param range_to_cluster: python list of values to cluster e.g. for classification [3,4,5] for the three types of veg
    :param distance: how close must two points be to be put in the same cluster
    :param min_pts: minimum number of points each point must have in a radius of size "distance"
    :param cluster_attribute: the name given to the clustering labels
    :param minimum_length: the minimum length of a cluster
    """
    # we shouldn't use las_modules.cluster function because it acts on a file, not on a family of points
    x = infile.x
    y = infile.y
    z = infile.z
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions
    coords = np.stack((x, y, z), axis=1)
    # make the clusters
    attr = infile.reader.get_dimension(attribute)
    mask = uicondition2mask(range_to_cluster)(attr)
    labels = 1 + clustering(coords[mask], distance, minimum_length, min_pts)
    # assign the target classification's labels
    labels_allpts[mask] = labels  # find our labels (DBSCAN starts at -1 and we want to start at 0, so add 1)

    outfile.writer.set_dimension(cluster_attribute, labels_allpts)


def eigencluster_labels_v01_0(infile,
                              outfile,
                              classification_to_cluster,
                              tolerance,
                              min_pts,
                              cluster_attribute,
                              eigenvector_number,
                              minimum_length):
    """
    Inputs a file and a classification to cluster. Outputs a file with cluster labels.
    Clusters with label 0 are non-core points, i.e. points without "min_pts" within
    "tolerance" (see DBSCAN documentation), or points outside the classification to cluster.
    :param infile: input file name
    :param outfile: output file name
    :param classification_to_cluster: which points do we want to cluster
    :param tolerance: see min_pts
    :param min_pts: minimum number of points each point must have in a radius of size "tolerance"
    :param cluster_attribute: the name given to the clustering labels
    :param eigenvector: 0, 1 or 2
    :return:
    """
    # we shouldn't use las_modules.cluster function because it acts on a file, not on a family of points
    infile = File(infile, mode="r")
    x = infile.x
    y = infile.y
    z = infile.z

    # extract the componenets of the desired eigenvector
    v0 = infile.reader.get_dimension(f"eig{eigenvector_number}0")
    v1 = infile.reader.get_dimension(f"eig{eigenvector_number}1")
    v2 = infile.reader.get_dimension(f"eig{eigenvector_number}2")
    eigenvector = np.stack((v0, v1, v2), axis=1)
    classn = infile.classification
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions

    coords = np.stack((x, y, z), axis=1)
    # make the cluster labels
    labels = 1 + eigen_clustering(coords[classn == classification_to_cluster],
                                  eigenvector[classn == classification_to_cluster],
                                  tolerance,
                                  5,
                                  minimum_length,
                                  min_pts)
    # assign the target classification's labels
    labels_allpts[classn == classification_to_cluster] = labels
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


def eigencluster_labels_v01_1(infile,
                              outfile,
                              eigenvector_number,
                              attribute,
                              range_to_cluster,
                              distance,
                              min_pts,
                              cluster_attribute,
                              minimum_length):
    """
    Input a file and select an eigenvector and attribute range to cluster. Outputs a file with cluster labels.
    Clusters with label 0 are non-core points, i.e. points without "min_pts" within
    "distance" (see DBSCAN documentation), or points outside the classification to cluster.
    :param infile: input file name
    :param outfile: output file name
    :param eigenvector_number: 0, 1 or 2
    :param attribute: the attribute which you want to use to select a range from
    :param range_to_cluster: python list of values to cluster e.g. for classification [3,4,5] for the three types of veg
    :param distance: how close must two points be to be put in the same cluster
    :param min_pts: minimum number of points each point must have in a radius of size "distance"
    :param cluster_attribute: the name given to the clustering labels
    :param minimum_length: the minimum length of a cluster
    :return:
    """
    # we shouldn't use las_modules.cluster function because it acts on a file, not on a family of points
    infile = File(infile, mode="r")
    x = infile.x
    y = infile.y
    z = infile.z

    # extract the componenets of the desired eigenvector
    v0 = infile.reader.get_dimension(f"eig{eigenvector_number}0")
    v1 = infile.reader.get_dimension(f"eig{eigenvector_number}1")
    v2 = infile.reader.get_dimension(f"eig{eigenvector_number}2")
    eigenvector = np.stack((v0, v1, v2), axis=1)
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions

    coords = np.stack((x, y, z), axis=1)
    # make the cluster labels
    mask = np.zeros(len(infile), dtype=bool)
    for item in range_to_cluster:
        mask[(infile.reader.get_dimension(attribute) >= item[0]) & (
                infile.reader.get_dimension(attribute) <= item[-1])] = True
    labels = 1 + eigen_clustering(coords[mask],
                                  eigenvector[mask],
                                  distance,
                                  5,
                                  minimum_length,
                                  min_pts)
    # assign the target classification's labels
    labels_allpts[mask] = labels
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


def eigencluster_labels_v01_2(infile,
                              outfile,
                              eigenvector_number,
                              attribute,
                              range_to_cluster,
                              distance,
                              min_pts,
                              cluster_attribute,
                              minimum_length):
    """
    Input a file and select an eigenvector and attribute range to cluster. Outputs a file with cluster labels.
    Clusters with label 0 are non-core points, i.e. points without "min_pts" within
    "distance" (see DBSCAN documentation), or points outside the classification to cluster.
    :param infile: input file name
    :param outfile: output file name
    :param eigenvector_number: 0, 1 or 2
    :param attribute: the attribute which you want to use to select a range from
    :param range_to_cluster: python list of values to cluster e.g. for classification [3,4,5] for the three types of veg
    :param distance: how close must two points be to be put in the same cluster
    :param min_pts: minimum number of points each point must have in a radius of size "distance"
    :param cluster_attribute: the name given to the clustering labels
    :param minimum_length: the minimum length of a cluster
    :return:
    """
    # we shouldn't use las_modules.cluster function because it acts on a file, not on a family of points
    infile = File(infile, mode="r")
    x = infile.x
    y = infile.y
    z = infile.z

    # extract the componenets of the desired eigenvector
    v0 = infile.reader.get_dimension(f"eig{eigenvector_number}0")
    v1 = infile.reader.get_dimension(f"eig{eigenvector_number}1")
    v2 = infile.reader.get_dimension(f"eig{eigenvector_number}2")
    eigenvector = np.stack((v0, v1, v2), axis=1)
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions

    coords = np.stack((x, y, z), axis=1)
    # make the cluster labels
    attr = infile.reader.get_dimension(attribute)
    mask = uicondition2mask(range_to_cluster)(attr)
    labels = 1 + eigen_clustering(coords[mask],
                                  eigenvector[mask],
                                  distance,
                                  5,
                                  minimum_length,
                                  min_pts)
    # assign the target classification's labels
    labels_allpts[mask] = labels
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


def eigencluster_labels_v01_3(infile,
                              outfile,
                              eigenvector_number,
                              attribute,
                              range_to_cluster,
                              distance,
                              min_pts,
                              cluster_attribute,
                              minimum_length):
    """
    Input a file and select an eigenvector and attribute range to cluster. Outputs a file with cluster labels.
    Clusters with label 0 are non-core points, i.e. points without "min_pts" within
    "distance" (see DBSCAN documentation), or points outside the classification to cluster.
    :param infile: input file name
    :param outfile: output file name
    :param eigenvector_number: 0, 1 or 2
    :param attribute: the attribute which you want to use to select a range from
    :param range_to_cluster: python list of values to cluster e.g. for classification [3,4,5] for the three types of veg
    :param distance: how close must two points be to be put in the same cluster
    :param min_pts: minimum number of points each point must have in a radius of size "distance"
    :param cluster_attribute: the name given to the clustering labels
    :param minimum_length: the minimum length of a cluster
    :return:
    """
    # we shouldn't use las_modules.cluster function because it acts on a file, not on a family of points

    x = infile.x
    y = infile.y
    z = infile.z

    # extract the componenets of the desired eigenvector
    v0 = infile.reader.get_dimension(f"eig{eigenvector_number}0")
    v1 = infile.reader.get_dimension(f"eig{eigenvector_number}1")
    v2 = infile.reader.get_dimension(f"eig{eigenvector_number}2")
    eigenvector = np.stack((v0, v1, v2), axis=1)
    # make a vector to store labels
    labels_allpts = np.zeros(len(infile), dtype=int)
    # get the point positions

    coords = np.stack((x, y, z), axis=1)
    # make the cluster labels
    attr = infile.reader.get_dimension(attribute)
    mask = uicondition2mask(range_to_cluster)(attr)
    labels = 1 + eigen_clustering(coords[mask],
                                  eigenvector[mask],
                                  distance,
                                  5,
                                  minimum_length,
                                  min_pts)
    # assign the target classification's labels
    labels_allpts[mask] = labels

    outfile.writer.set_dimension(cluster_attribute, labels_allpts)


def count_v01_0(tile,
                output_file,
                attribute):
    """
    adds a number to each point reflecting the number of points with the same value for chosen attribute
    :param tile: an input tile
    :param output_file: name for output tile
    :param attribute: attribute to be counted
    :return:
    """
    # read the file and make the new one
    inFile = File(tile, mode="r")
    outfile = File(output_file, mode="w", header=inFile.header)
    dimensions = [spec.name for spec in inFile.point_format]
    # add in the new count attribute
    if attribute + "count" not in dimensions:
        outfile.define_new_dimension(name=attribute + "count", data_type=5, description=attribute + "count")
    # add pre-existing point records
    for dimension in dimensions:
        if dimension != attribute + "count":
            dat = inFile.reader.get_dimension(dimension)
            outfile.writer.set_dimension(dimension, dat)
    # count the attribute using numpy unique
    unq, inv, cnt = np.unique(outfile.reader.get_dimension(attribute), return_index=False, return_inverse=True,
                              return_counts=True)
    # set the counts to the new attribute
    outfile.writer.set_dimension(attribute + "count", cnt[inv])
    outfile.close()


def ferry_v01_0(infile, outfile, attribute1, attribute2, renumber, start=0):
    """
    :param infile: file name to read
    :param outfile: file name to write
    :param attribute1: attribute whose values will be inserted into attribute2
    :param attribute2: attribute to be overwritten by attribute1
    :param renumber: True/False to renumber as consecutive integers
    :param start: what to renumber from (ignored if renumber is False)
    """
    inFile = File(infile)
    outFile = File(outfile, mode="w", header=inFile.header)
    outFile.points = inFile.points
    a = inFile.reader.get_dimension(attribute1)
    if renumber:
        unq, ind, inv = np.unique(a, return_index=True, return_inverse=True, return_counts=False)
        a = np.arange(ind.size)[inv] + start
    outFile.writer.set_dimension(attribute2, a)


def ferry_v01_1(infile, outfile, attribute1, attribute2, renumber, start=0, manipulate=lambda x: x):
    """
    :param infile: file name to read
    :param outfile: file name to write
    :param attribute1: attribute whose values will be inserted into attribute2
    :param attribute2: attribute to be overwritten by attribute1
    :param renumber: True/False to renumber as consecutive integers
    :param start: what to renumber from (ignored if renumber is False)
    """
    inFile = File(infile)
    outFile = File(outfile, mode="w", header=inFile.header)
    outFile.points = inFile.points
    a = inFile.reader.get_dimension(attribute1)
    if renumber:
        unq, ind, inv = np.unique(a, return_index=True, return_inverse=True, return_counts=False)
        a = np.arange(ind.size)[inv] + start
    outFile.writer.set_dimension(attribute2, manipulate(a))


def decimate_v01_0(infile, outfile, decimated_outfile, voxel_size, inverter_attribute):
    """
    Produces two new files. One is decimated with no extra attributes, another is not decimated which holds an attribute
    to recover info from decimated file.
    :param infile: file to decimate
    :param outfile: the file with inverter attibute
    :param decimated_outfile: the decimated file with fewer points
    :param voxel_size: voxel side length
    :param inverter_attribute: name of inverter attribute
    :return: two files, one with inverter, one with fewer points
    """
    inFile = File(infile)
    outFile = File(outfile, mode="w", header=inFile.header)
    u = voxel_size
    x = inFile.x
    y = inFile.y
    z = inFile.z
    unq, ind, inv = np.unique(np.floor(np.stack((x, y, z), axis=1) / u).astype(int), return_index=True,
                              return_inverse=True, return_counts=False, axis=0)
    dimensions = [spec.name for spec in inFile.point_format]
    # add dimension
    outFile.define_new_dimension(name=inverter_attribute, data_type=5, description="inverses for decimation")
    # add pre-existing point records
    for dimension in dimensions:
        dat = inFile.reader.get_dimension(dimension)
        outFile.writer.set_dimension(dimension, dat)
    # add new dimension
    outFile.writer.set_dimension(inverter_attribute, inv)
    outFile.close()
    decFile = File(decimated_outfile, mode="w", header=inFile.header)
    dimensions = [spec.name for spec in inFile.point_format]
    # add pre-existing point records
    for dimension in dimensions:
        dat = inFile.reader.get_dimension(dimension)
        decFile.writer.set_dimension(dimension, dat[ind])
    # add new x, y, z
    decFile.x = u * unq[:, 0]
    decFile.y = u * unq[:, 1]
    decFile.z = u * unq[:, 2]
    decFile.close()


def undecimate_v01_0(infile_with_inv, infile_decimated, outfile, inverter_attribute, attributes_to_copy):
    """
    reverses decimation
    :param infile_with_inv: file with inverter attribute
    :param infile_decimated: decimated file with fewer points
    :param outfile: new file with attributes recovered from decimation
    :param inverter_attribute: name of inverter attribute on infile_with_inv
    :param attributes_to_copy: attributes for recovery
    :return:
    """
    inFile1 = File(infile_with_inv)
    inFile2 = File(infile_decimated)
    outFile = File(outfile, mode="w", header=inFile2.header)
    dimensions1 = [spec.name for spec in inFile1.point_format]
    dimensions2 = [spec.name for spec in inFile2.point_format]
    # add pre-existing point records
    inv = inFile1.reader.get_dimension(inverter_attribute)
    for dimension in dimensions2:
        if dimension in dimensions1:
            dat = inFile1.reader.get_dimension(dimension)
            outFile.writer.set_dimension(dimension, dat)
    for dimension in attributes_to_copy:
        dat = inFile2.reader.get_dimension(dimension)
        outFile.writer.set_dimension(dimension, dat[inv])
    outFile.close()


def virus_v01_0(infile, outfile, distance, num_itter, virus_attribute, virus_range, select_attribute, select_range,
                protect_attribute, protect_range, attack_attribute, value):
    """
    Used to spread an attribute to nearby points of selected points. OR we can put a number in value to reset the
    selected points' attack attribute to that number
    :param infile:
    :param outfile:
    :param virus_attribute: Attribute to range in order to decide which points to `spread out from'.
    :param virus_range: range for virus_attribute
    :param select_attribute: An attribute used to range for selecting affected points.
    :param select_range: Range for select_attribute.
    :param protect_attribute: An attribute used to range for protecting points.
    :param protect_range:Range for protect_attribute.
    :param attack_attribute:
    :param value: Values to which we should change the attack_attribute OR name of attribute to spread.
    :return:
    """
    inFile = File(infile)
    vir = inFile.reader.get_dimension(virus_attribute)
    if select_attribute is not None:
        sel = inFile.reader.get_dimension(select_attribute)
        mask3 = uicondition2mask(select_range)(sel)
    else:
        mask3 = np.ones(len(inFile), dtype=bool)
    if protect_attribute is not None:
        ptc = inFile.reader.get_dimension(protect_attribute)
        mask4 = uicondition2mask(protect_range)(ptc)
    else:
        mask4 = np.zeros(len(inFile), dtype=bool)
    mask1 = uicondition2mask(virus_range)(vir)
    mask2 = mask3 & (~ mask4)
    cls = virus_background(inFile, distance, num_itter, mask1, mask2, attack_attribute, value)
    outFile = File(outfile, mode="w", header=inFile.header)
    outFile.points = inFile.points
    outFile.writer.set_dimension(attack_attribute, cls)
    outFile.close()


def virus_v01_1(infile, outfile, distance, num_itter, virus_attribute, virus_range, select_attribute, select_range,
                protect_attribute, protect_range, attack_attribute, value):
    """
    Used to spread an attribute to nearby points of selected points. OR we can put a number in value to reset the
    selected points' attack attribute to that number
    :param infile:
    :param outfile:
    :param virus_attribute: Attribute to range in order to decide which points to `spread out from'.
    :param virus_range: range for virus_attribute
    :param select_attribute: An attribute used to range for selecting affected points.
    :param select_range: Range for select_attribute.
    :param protect_attribute: An attribute used to range for protecting points.
    :param protect_range:Range for protect_attribute.
    :param attack_attribute:
    :param value: Values to which we should change the attack_attribute OR name of attribute to spread.
    :return:
    """

    vir = infile.reader.get_dimension(virus_attribute)
    if select_attribute is not None:
        sel = infile.reader.get_dimension(select_attribute)
        mask3 = uicondition2mask(select_range)(sel)
    else:
        mask3 = np.ones(len(infile), dtype=bool)
    if protect_attribute is not None:
        ptc = infile.reader.get_dimension(protect_attribute)
        mask4 = uicondition2mask(protect_range)(ptc)
    else:
        mask4 = np.zeros(len(infile), dtype=bool)
    mask1 = uicondition2mask(virus_range)(vir)
    mask2 = mask3 & (~ mask4)
    cls = virus_background(infile, distance, num_itter, mask1, mask2, attack_attribute, value)
    outfile.writer.set_dimension(attack_attribute, cls)


def attributeupdate_v01_0(infile, outfile, select_attribute, select_range, protect_attribute, protect_range,
                          attack_attribute, value):
    """
    Note: I think this somewhat "pathes the way" for the way in which we should set out our modules. It gives the user:
    A select attribute to control which points will potentially be affected.
    A protect attribute to control which points will not be affected.
    An attack attribute which will be changed for the affected points.
    This module simply updates an attribute to the given value.
    :param infile:
    :param outfile:
    :param select_attribute: An attribute used to range for selecting affected points.
    :param select_range: Range for select_attribute.
    :param protect_attribute: An attribute used to range for protecting points.
    :param protect_range: Range for protect_attribute.
    :param attack_attribute: Attribute to be changed
    :param value: Values to which we should change the attack_attribute.
    :return:
    """
    inFile = File(infile)
    cls = 1 * inFile.reader.get_dimension(attack_attribute)
    if select_attribute is not None:
        sel = inFile.reader.get_dimension(select_attribute)
        mask1 = uicondition2mask(select_range)(sel)
    else:
        mask1 = np.ones(len(inFile), dtype=bool)
    if protect_attribute is not None:
        mask2 = uicondition2mask(protect_range)(inFile.reader.get_dimension(protect_attribute))
    else:
        mask2 = np.zeros(len(inFile), dtype=bool)
    cls[mask1 & (~mask2)] = value
    outFile = File(outfile, mode="w", header=inFile.header)
    outFile.points = inFile.points
    outFile.writer.set_dimension(attack_attribute, cls)
    outFile.close()


def attributeupdate_v01_1(infile, outfile, select_attribute, select_range, protect_attribute, protect_range,
                          attack_attribute, value):
    """
    Note: I think this somewhat "pathes the way" for the way in which we should set out our modules. It gives the user:
    A select attribute to control which points will potentially be affected.
    A protect attribute to control which points will not be affected.
    An attack attribute which will be changed for the affected points.
    This module simply updates an attribute to the given value.
    :param infile:
    :param outfile:
    :param select_attribute: An attribute used to range for selecting affected points.
    :param select_range: Range for select_attribute.
    :param protect_attribute: An attribute used to range for protecting points.
    :param protect_range: Range for protect_attribute.
    :param attack_attribute: Attribute to be changed
    :param value: Values to which we should change the attack_attribute.
    :return:
    """

    cls = 1 * infile.reader.get_dimension(attack_attribute)
    if select_attribute is not None:
        sel = infile.reader.get_dimension(select_attribute)
        mask1 = uicondition2mask(select_range)(sel)
    else:
        mask1 = np.ones(len(infile), dtype=bool)
    if protect_attribute is not None:
        mask2 = uicondition2mask(protect_range)(infile.reader.get_dimension(protect_attribute))
    else:
        mask2 = np.zeros(len(infile), dtype=bool)
    cls[mask1 & (~mask2)] = value

    outfile.writer.set_dimension(attack_attribute, cls)


def create_attributes(infile, output_file, attribute_names=None):
    """
    Create the extra dimensions that are putted into attribute_names.

    :param output_file: Name of the output file.
    :param infile: las file on which to create the attributes.
    :param attribute_names: Attributes names put like ("name","data_type","description")
    e.x. [("dimension1d2d3d",6,"dimension")]
    :return: output las file in write mode
    """
    if attribute_names is None:
        attribute_names = []

    out_file = File(output_file, mode="w", header=infile.header)
    dimensions = [spec.name for spec in infile.point_format]
    # add new dimension
    for dim in attribute_names:
        if dim[0] not in dimensions:
            out_file.define_new_dimension(name=dim[0], data_type=dim[1], description=dim[2])

    # add pre-existing point records
    for dimension in dimensions:
        dat = infile.reader.get_dimension(dimension)
        out_file.writer.set_dimension(dimension, dat)

    return out_file
