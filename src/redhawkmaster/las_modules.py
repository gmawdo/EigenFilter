from laspy.file import File
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
import os


def las_range(dimension, start=(-sys.maxsize - 1), end=sys.maxsize, reverse=False, point_id_mask=np.array([])):
    """
    Function that return range between start and end values on some dimension.
    The interval is [start,end) which means >=start and >end.

    :param dimension: Array of values on which we compute the range
    :type dimension: numpy array
    :param start: start of the range (default minimum integer value -sys.maxsize-1)
    :type start: float
    :param end: end of the range (default max integer value sys.maxsize)
    :type end: float
    :param reverse: flag which indicates if we take everything in the range or outside the range (default False;
     False in, True out)
    :type reverse: bool
    :param point_id_mask: numpy array of id  of the points in the dimension param (default np.array([]);
     if it is empty we get the bool mask of the range)
    :type point_id_mask: numpy aray
    :return  numpy array of point id if the point_id_mask isn't empty, bool array if it is
    """
    # Function which will range the specific tile
    # from the values start and end on a specific dimension
    # Reading the file
    # Please note the location of the file
    # inFile = File(tile_name, mode = 'r')

    # For each dimension we make a mask from start until end
    # If a dimension is in lowercase letters is scaled one
    dimension = dimension[point_id_mask]
    if point_id_mask.size == 0:
        mask = np.zeros(len(dimension), dtype=bool)
    else:
        mask = (dimension >= start) & (dimension < end)
    if reverse:
        mask = ~mask

    return point_id_mask[np.where(mask)[0]]


def duplicate_attr(infile, attribute_in, attribute_out, attr_descrp, attr_type):
    """
    Gets input laspy object and it makes another laspy from that object that
    has extra dimension with name as atrribute_out, description attr_descrp and data type as attr_type and
    same values as the attribute_in dimension from infile. Similar Ferry filter in PDAL.

    :param infile: laspy object from which to take the dimension we want to copy into.
    :type infile: laspy object
    :param attribute_in: Name of the attribute in lowercase
    :type attribute_in: string
    :param attribute_out: New name for the extra attribute
    :type attribute_out: string
    :param attr_descrp: Description of the new attribute
    :type attr_descrp: string
    :param attr_type: Data type of the extra attribute as laspy documentaion (see laspy documentation)
    :type attr_type: int
    :return  laspy object and writes a file with the same name as infile just added the name of the extra attr
    """
    inFile = infile

    # Temp file for creating the new dimension
    outFile = File("T000_extradim.las", mode="w", header=inFile.header)

    # Create the dimension
    outFile.define_new_dimension(name=attribute_out, data_type=attr_type, description=attr_descrp)

    # Populate the points
    for dimension in inFile.point_format:
        dat = inFile.reader.get_dimension(dimension.name)
        outFile.writer.set_dimension(dimension.name, dat)

    outFile.close()
    # Open the temp file
    inFile = File("T000_extradim.las", mode="r")

    # Open output file with same name as infile
    outFile1 = File(infile.filename.split('/')[-1].split('.las')[0] + ".las",
                    mode="w", header=inFile.header)

    # Populate the points
    for dimension in inFile.point_format:
        dat = inFile.reader.get_dimension(dimension.name)
        outFile1.writer.set_dimension(dimension.name, dat)

    # Get the dimension that you want to copy
    in_spec = inFile.reader.get_dimension(attribute_in)
    # Copy it in the new dimension
    outFile1.writer.set_dimension(attribute_out, in_spec)
    # Delet the temp file
    os.system('rm T000_extradim.las')

    return outFile1


def flightline_point_counter(infile, clip, nh, mask):
    """
    Count the points around the flight line of the las file.

    :param infile: laspy object on which to change the intensity
    :type infile: laspy object
    :param clip: distances from where to cut the distances of the neighbour points to
    :type clip: float
    :param nh: number of neighbours of point
    :type nh: int
    :param mask: point id mask or bool mask for which points to be changed of the intensity
    :return  it changes the intensity of the infile
    """
    # Input class 10 data only

    # Pass1
    # "radius": 1.0,
    # "min_k": 40,

    # Pass2
    # "radius": 2.0,
    # "min_k": 160,

    # X(m) radius - for changing the radius change the variable clip

    classification = infile.Classification[mask]
    x_array = infile.x[mask]
    y_array = infile.y[mask]
    z_array = infile.z[mask]

    # We measure two sets of point values positions with each other
    # We read in one set 'Classification'
    # We duplicate a second Classification from the 1st set of points
    # Example, here we read in Classification 10 and duplicate a set called class99
    class99 = (classification == 10)

    # XYZ dataset
    # np.vstack = Stack arrays in sequence vertically (row wise).
    coords = np.vstack((x_array, y_array, z_array))

    # Two XYZ dataset with all classification 99
    # NOT SURE WHAT THIS DOES
    coords_flight = np.vstack((x_array[class99], y_array[class99], z_array[class99]))

    # n_neighbors is the number of neighbors of a point, I can't take everything within 1m
    # but I can take a lot of them for now 125
    if len(x_array[class99]) == 0:
        print("The dataset is empty.")
        return infile

    if len(x_array[class99]) < nh:
        nh = len(x_array[class99])

    try:
        nhbrs = NearestNeighbors(n_neighbors=nh, algorithm="kd_tree").fit(np.transpose(coords_flight))
        distances, indices = nhbrs.kneighbors(np.transpose(coords))

        # Count number of points within the cluster
        intensity = infile.Intensity[mask]

        intensity_array = intensity[indices].astype(float)
        intensity_array[distances >= clip] = np.nan
        intensity_count = np.sum(distances < clip, axis=1)

        # Apply the median intensity
        intensity = intensity_count.astype('uint16')

        tmp_int = infile.intensity
        tmp_int[mask] = intensity
        infile.intensity = tmp_int
        # Get the changed intensity to output
        return intensity
    except ValueError as e:
        print("The dataset for mild denoise on flightline is empty")


def virus(infile, clip, num_itter, classif):
    """
    It is spreading the classification classif using some clip on the infile with
    some number iterations.

    :param infile: laspy object on which to change the intensity
    :type infile: laspy object
    :param clip: distances from where to cut the distances of the neighbour points to
    :type clip: float
    :param num_itter: number of iterations means how much the the classification will spread
    :type num_itter: int
    :param classif: what classification will be put on a points
    :type classif: int
    :return  numpy array of the new classification
    """

    # Coords of the file along with the classification
    cls = infile.Classification
    x_array = infile.x
    y_array = infile.y
    z_array = infile.z

    coords = np.vstack((x_array, y_array, z_array))

    # Loop for the iterations
    for i in range(0, num_itter):

        # Extract the classification in a rray
        class1 = (cls == classif)

        # Pull the coordinates of that classification
        coords_flight = np.vstack((x_array[class1], y_array[class1], z_array[class1]))

        # If we don't have zero points proceed
        if len(coords_flight[0]) != 0:
            # Compute the distances of the first neighbour of every point
            nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(np.transpose(coords_flight))
            distances, indices = nhbrs.kneighbors(np.transpose(coords))

            # The distances that are less then the clip get classified
            cls[np.logical_and(distances[:, 0] < clip, np.logical_not(class1))] = classif

    infile.classification = cls

    return cls


def virus_background(infile, clip, num_itter, mask1, mask2, attribute_attack, value):
    """
    It is spreading the classification classif using some clip on the infile with
    some number iterations.

    :param infile: laspy object on which to change the intensity
    :type infile: laspy object
    :param clip: distances from where to cut the distances of the neighbour points to
    :type clip: float
    :param num_itter: number of iterations means how much the the classification will spread
    :type num_itter: int
    :param mask1: describing points which we want to virus out of
    :type mask1: numpy array (dtype == bool)
    :param mask2: describing points we want to relabel
    :type mask2: numpy array (dtype == bool)
    :param classif: what classification will be put on a points
    :type classif: int
    :return  numpy array of the new classification
    """

    # Coords of the file along with the classification
    cls = infile.reader.get_dimension(attribute_attack)
    x_array = infile.x
    y_array = infile.y
    z_array = infile.z

    coords = np.stack((x_array, y_array, z_array), axis = 1)

    # Loop for the iterations
    for i in range(num_itter):
        # If we don't have zero points proceed
        if mask1.any() and mask2.any():
            # Compute the distances of the first neighbour of every point
            nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coords[mask1])
            distances, indices = nhbrs.kneighbors(coords[mask2])
            cls2 = cls[mask2]
            # The distances that are less then the clip get classified
            cls2[distances[:, 0] < clip) = value
            cls[mask2] = cls2

    return cls


def rh_clip(infile, clip=100, cls_int=10, point_id_mask=np.array([])):
    """
    Classifying a noise on 2D space around some classification.

    :param infile: laspy object on which to change the classification
    :type infile: laspy object
    :param clip: distances how much to classify as noise. (default 100 in meters)
    :type clip: float
    :param cls_int: around which classification to classify (default 10 (flight line))
    :type cls_int: int
    :param point_id_mask: mask to classify on (if we want part of the input file)
    :type point_id_mask: numpy array
    :return  numpy array of the new classification
    """

    # Coords XY and classification arrays
    classification = infile.Classification[point_id_mask]
    x_array = infile.x[point_id_mask]
    y_array = infile.y[point_id_mask]

    # Extract the classification
    class10 = (classification == cls_int)

    coords = np.vstack((x_array, y_array))

    coords_flight = np.vstack((x_array[class10], y_array[class10]))

    # If we don't have that classification finish the script
    if len(x_array[class10]) == 0:
        print("There is not flightline in this dataset.")
        return 0

    # Compute the distances
    nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(np.transpose(coords_flight))
    distances, indices = nhbrs.kneighbors(np.transpose(coords))

    # Get the clip
    mask = np.logical_and(distances[:, 0] < clip, np.logical_not(class10))

    # Get the mask of the clip
    cls_mask = np.logical_and(np.logical_not(mask), np.logical_not(class10))

    # Classify it as class 7 (noise)
    classification[cls_mask] = 7

    # Put the new classification into the file
    cls_tmp = infile.classification
    cls_tmp[point_id_mask] = classification
    infile.classification = cls_tmp

    return classification


def rh_kdistance(infile, output_file_name='', k=10, make_dimension=True, mask=np.array([])):
    """
    Compute kd distance on a file.

    :param mask:
    :param infile: laspy object on which to create kd distance and populate it, or just populate it
    :type infile: laspy object
    :param output_file_name: output file name (default '')
    :type output_file_name: string
    :param k: which k neighbour distance to output (default k)
    :type k: int
    :param make_dimension: to make kd_distance dimension or not to make (default True)
    :type make_dimension: bool
    :return  file with kd_distance or populate kd_distance dimension
    """

    # The first kdistance is the point itself so we need to put + 1
    k = k + 1

    # coords of the file
    coords = np.vstack((infile.x[mask], infile.y[mask], infile.z[mask]))

    # Compute the distances
    nhbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(np.transpose(coords))
    distances, indices = nhbrs.kneighbors(np.transpose(coords))

    if make_dimension:

        # Make the file with kdistance
        outFile = File(output_file_name, mode="w", header=infile.header)
        outFile.define_new_dimension(name="kdistance", data_type=10, description="KDistance")

        # Populate the points
        for dimension in infile.point_format:
            dat = infile.reader.get_dimension(dimension.name)
            outFile.writer.set_dimension(dimension.name, dat)

        # Populate the kdistance
        outFile.writer.set_dimension("kdistance", distances[:, -1])
        return outFile
    else:
        # if it is already made just populate it
        infile.kdistance[mask] = distances[:, -1]


def rh_assign(dimension, value, mask):
    """
    Assign filter from PDAL.

    :param dimension: array from which to assign the value to
    :type dimension: numpy array
    :param value: integer value of what to assign
    :type value: int
    :param mask: mask of which points to assign
    :type mask: numpy array or bool array
    :return  dimensions with assigned values
    """
    cls = dimension
    cls[mask] = value
    return cls


def rh_attribute_compute(infile, outname, k=50, radius=0.50, thresh=0.001, spacetime=True, v_speed=2,
                         decimate=False, u=0.1, N=6):
    """
    Computing different mathematical attributes.

    :param infile: input las file
    :type infile: laspy object
    :param outname: output file name
    :type outname: string
    :param k: number of neighbours (default 50)
    :type k: int
    :param radius: maximum radius of neighbourhoods
    :type radius: float
    :param thresh: threshold for non-zero eigenvalue (affects rank) (default 0.001)
    :type thresh: float
    :param spacetime: do you want spacetime? (default True)
    :type spacetime: bool
    :param v_speed: scale of spacetime solution (default 2)
    :type v_speed: float
    :param decimate: do you want to decimate? (default False)
    :type decimate: bool
    :param u: scale of decimation (e.g. 0.1 = 10cm) (default 0.1)
    :type u: float
    :param N:  number of subtiles (no edge effects) (default 6)
    :type N: int
    :return: las file to the system defined by outname
    """

    import numpy.linalg as LA

    outFile = File(outname, header=infile.header, mode='w')

    Specs = [spec.name for spec in infile.point_format]

    extra_dims = [
        ["xy_lin_reg", 9, "Linear regression."],
        ["lin_reg", 9, "Linear regression."],
        ["plan_reg", 9, "Planar regression."],
        ["eig0", 9, "Eigenvalue 0. "],
        ["eig1", 9, "Eigenvalue 1. "],
        ["eig2", 9, "Eigenvalue 2. "],
        ["rank", 5, "Structure matrix."],
        ["curv", 9, "Curvature."],
        ["iso", 9, "Isotropy"],
        ["ent", 9, "Entropy"],
        ["plang", 9, "Planar angle"],
        ["lang", 9, "Linear angle"],
        ["linearity", 9, "Closeness to a line"],
        ["planarity", 9, "Closeness to a plane"],
        ["scattering", 9, "3d-ness"],
        ["dim1_mw", 9, "Similar to linearity"],
        ["dim2_mw", 9, "Similar to planarity"],
        ["dim3_mw", 9, "Similar to scattering"],
        ["dim1_sd", 9, "Similar to linearity"],
        ["dim2_sd", 9, "Similar to planarity"],
        ["dim3_sd", 9, "Similar to scattering"],
        ["dimension", 5, "Dimension of nbhd"]
    ]

    for dim in extra_dims:
        if not (dim[0] in Specs):
            outFile.define_new_dimension(name=dim[0], data_type=dim[1], description=dim[2])

    for dimension in infile.point_format:
        dat = infile.reader.get_dimension(dimension.name)
        outFile.writer.set_dimension(dimension.name, dat)

    if v_speed == 0:
        spacetime = False

    x_array = infile.x
    y_array = infile.y
    z_array = infile.z
    heightaboveground = infile.heightaboveground
    gps_time = infile.gps_time
    intensity = infile.intensity
    classification = infile.classification

    dummy = 0 * x_array

    dummies = {'xy_lin_reg': dummy.astype(x_array.dtype), 'lin_reg': dummy.astype(x_array.dtype),
               'plan_reg': dummy.astype(x_array.dtype), 'eig0': dummy.astype(x_array.dtype),
               'eig1': dummy.astype(x_array.dtype), 'eig2': dummy.astype(x_array.dtype),
               'curv': dummy.astype(x_array.dtype), 'iso': dummy.astype(x_array.dtype),
               'rank': dummy.astype(classification.dtype),
               'ent': dummy.astype(x_array.dtype), 'plang': dummy.astype(x_array.dtype),
               'lang': dummy.astype(x_array.dtype), 'linearity': dummy.astype(x_array.dtype),
               'planarity': dummy.astype(x_array.dtype), 'scattering': dummy.astype(x_array.dtype),
               'dim1_mw': dummy.astype(x_array.dtype), 'dim2_mw': dummy.astype(x_array.dtype),
               'dim3_mw': dummy.astype(x_array.dtype), 'dim1_sd': dummy.astype(x_array.dtype),
               'dim2_sd': dummy.astype(x_array.dtype), 'dim3_sd': dummy.astype(x_array.dtype),
               'dimension': dummy.astype(x_array.dtype)}

    if len(x_array) != 0:

        times = list([np.quantile(gps_time, q=float(i) / float(N)) for i in range(N + 1)])

        for i in range(N):

            time_range = (times[i] <= gps_time) * (gps_time <= times[i + 1])

            coords = np.vstack((x_array[time_range], y_array[time_range],
                                heightaboveground[time_range]) + spacetime * (v_speed * gps_time[time_range],))

            if decimate:
                spatial_coords, ind, inv, cnt = np.unique(np.floor(coords[0:4, :] / u),
                                                          return_index=True, return_inverse=True,
                                                          return_counts=True, axis=1)
            else:
                ind = np.arange(len(x_array[time_range]))
                inv = ind

            coords = coords[:, ind]

            distances, indices = NearestNeighbors(n_neighbors=k, algorithm="kd_tree"). \
                fit(np.transpose(coords)). \
                kneighbors(np.transpose(coords))

            neighbours = coords[:, indices]
            keeping = distances < radius
            Ns = np.sum(keeping, axis=1)

            means = coords[:, :, None]
            raw_deviations = keeping * (neighbours - means) / np.sqrt(Ns[None, :, None])  # (3,N,k)
            cov_matrices = np.matmul(raw_deviations.transpose(1, 0, 2), raw_deviations.transpose(1, 2, 0))  # (N,3,3)
            cov_matrices = np.maximum(cov_matrices, cov_matrices.swapaxes(-1, -2))
            xy_covs = cov_matrices[:, 0, 1]
            yz_covs = cov_matrices[:, 1, 2]
            zx_covs = cov_matrices[:, 2, 0]
            xx_covs = cov_matrices[:, 0, 0]
            yy_covs = cov_matrices[:, 1, 1]
            zz_covs = cov_matrices[:, 2, 2]

            evals, evects = LA.eigh(cov_matrices)

            exp1 = xx_covs * yy_covs * zz_covs + 2 * xy_covs * yz_covs * zx_covs
            exp2 = xx_covs * yz_covs * yz_covs + yy_covs * zx_covs * zx_covs + zz_covs * xy_covs * xy_covs
            xy_lin_regs = abs(xy_covs / np.sqrt(xx_covs * yy_covs))

            plan_regs = exp2 / exp1

            xy_lin_regs[np.logical_or(np.isnan(xy_lin_regs), np.isinf(xy_lin_regs))] = 1

            lin_regs = abs(xy_covs * yz_covs * zx_covs / (xx_covs * yy_covs * zz_covs))
            lin_regs[np.logical_or(np.isnan(lin_regs), np.isinf(lin_regs))] = 1

            plan_regs[np.logical_or(np.isnan(plan_regs), np.isinf(plan_regs))] = 1

            ranks = np.sum(evals > thresh, axis=1, dtype=np.double)
            means = np.mean(neighbours, axis=2)

            # normal curvature filters
            p0 = evals[:, -3] / (evals[:, -1] + evals[:, -2] + evals[:, -3])
            p1 = evals[:, -2] / (evals[:, -1] + evals[:, -2] + evals[:, -3])
            p2 = evals[:, -1] / (evals[:, -1] + evals[:, -2] + evals[:, -3])

            p0 = -p0 * np.log(p0)
            p1 = -p1 * np.log(p1)
            p2 = -p2 * np.log(p2)
            p0[np.isnan(p0)] = 0
            p1[np.isnan(p1)] = 0
            p2[np.isnan(p2)] = 0

            E = (p0 + p1 + p2) / np.log(3)

            if not decimate:
                cnt = 3 * k / (4 * np.pi * (distances[:, -1] ** 3))

            dummies['xy_lin_reg'][time_range] = xy_lin_regs[inv].astype(x_array.dtype)
            dummies['lin_reg'][time_range] = lin_regs[inv].astype(x_array.dtype)
            dummies['plan_reg'][time_range] = plan_regs[inv].astype(x_array.dtype)
            dummies['eig0'][time_range] = evals[:, -3][inv].astype(x_array.dtype)
            dummies['eig1'][time_range] = evals[:, -2][inv].astype(x_array.dtype)
            dummies['eig2'][time_range] = evals[:, -1][inv].astype(x_array.dtype)
            dummies['curv'][time_range] = p0[inv].astype(x_array.dtype)

            dummies['curv'][np.logical_or(np.isnan(dummies['curv']), np.isinf(dummies['curv']))] = (
                    0 * x_array[np.logical_or(np.isnan(dummies['curv']), np.isinf(dummies['curv']))]).astype(
                x_array.dtype)

            dummies['iso'][time_range] = ((evals[:, -1] + evals[:, -2] + evals[:, -3]) / np.sqrt(
                3 * (evals[:, -1] ** 2 + evals[:, -2] ** 2 + evals[:, -3] ** 2)))[inv].astype(x_array.dtype)

            dummies['iso'][np.logical_or(np.isnan(dummies['iso']), np.isinf(dummies['iso']))] = (
                    0 * x_array[np.logical_or(np.isnan(dummies['iso']), np.isinf(dummies['iso']))]).astype(
                x_array.dtype)

            dummies['rank'][time_range] = ranks[inv].astype(classification.dtype)

            # dummies['impdec'][time_range] = cnt[inv].astype(x_array.dtype)

            dummies['ent'][time_range] = E[inv].astype(x_array.dtype)

            dummies['plang'][time_range] = np.clip(2 * (np.arccos(abs(evects[:, 2, -3]) / (
                np.sqrt(evects[:, 2, -3] ** 2 + evects[:, 1, -3] ** 2 + evects[:, 0, -3] ** 2))) / np.pi), 0, 1)[
                inv].astype(x_array.dtype)

            dummies['plang'][np.logical_or(np.isnan(dummies['plang']), np.isinf(dummies['plang']))] = (
                    0 * x_array[np.logical_or(np.isnan(dummies['plang']), np.isinf(dummies['plang']))]).astype(
                x_array.dtype)

            dummies['lang'][time_range] = np.clip(2 * (np.arccos(abs(evects[:, 2, -1]) / (
                np.sqrt(evects[:, 2, -1] ** 2 + evects[:, 1, -1] ** 2 + evects[:, 0, -1] ** 2))) / np.pi), 0, 1)[
                inv].astype(x_array.dtype)

            dummies['lang'][np.logical_or(np.isnan(dummies['lang']), np.isinf(dummies['lang']))] = (
                    0 * x_array[np.logical_or(np.isnan(dummies['lang']), np.isinf(dummies['lang']))]).astype(
                x_array.dtype)

        some_definitions = {
            "dim1_mw": (lambda x, y, z: (z - y) / z),
            "dim2_mw": (lambda x, y, z: (y - x) / z),
            "dim3_mw": (lambda x, y, z: x / z),
            "dim1_sd": (lambda x, y, z: (z - y) / (x + y + z)),
            "dim2_sd": (lambda x, y, z: 2 * (y - x) / (x + y + z)),
            "dim3_sd": (lambda x, y, z: 3 * x / (x + y + z)),
            "dimension": (lambda x, y, z: 1 + np.argmax(
                np.stack(((z - y) / (x + y + z), 2 * (y - x) / (x + y + z), 3 * x / (x + y + z)), axis=1), axis=1)),
        }

        for function_lambdas in some_definitions:
            dummies[function_lambdas] = some_definitions[function_lambdas](dummies['eig0'], dummies['eig1'],
                                                                           dummies['eig2'])

            dummies[function_lambdas][
                np.logical_or(np.isnan(dummies[function_lambdas]), np.isinf(dummies[function_lambdas]))] = (0 * x_array[
                np.logical_or(np.isnan(dummies[function_lambdas]), np.isinf(dummies[function_lambdas]))]).astype(
                x_array.dtype)

        for signal in dummies:
            outFile.writer.set_dimension(signal, dummies[signal])
    else:
        for signal in dummies:
            outFile.writer.set_dimension(signal, np.zeros((1, len(x_array)), dtype=np.double))

        print('Empty dataset.')

    return outFile


def rh_mult_attr(infile, mul=1000):
    """
    Multiply the attr by some number.

    :param infile: las file on which to multiply the attributes.
    :type infile: laspy object
    :param mul: by how much to miltiply
    :return:
    """

    xy_lin_reg = infile.xy_lin_reg * mul
    plan_reg = infile.plan_reg * mul
    eig0 = infile.eig0 * mul
    eig1 = infile.eig1 * mul
    eig2 = infile.eig2 * mul
    rank = infile.rank * 1
    lin_reg = infile.lin_reg * mul
    curv = infile.curv * mul
    iso = infile.iso * mul
    ent = infile.ent * mul
    plang = infile.plang * mul
    lang = infile.lang * mul

    infile.xy_lin_reg = xy_lin_reg
    infile.plan_reg = plan_reg
    infile.eig0 = eig0
    infile.eig1 = eig1
    infile.eig2 = eig2
    infile.rank = rank
    infile.lin_reg = lin_reg
    infile.curv = curv
    infile.iso = iso
    infile.ent = ent
    infile.plang = plang
    infile.lang = lang


def rh_return_index(infile, outname=''):
    """
    Combine return info in one header.

    :param infile: las file from which to read return number and number of returns
    :param outname: name of the las file which will be outputed
    :return: laspy object
    """

    Specs = [spec.name for spec in infile.point_format]

    if not ("user_return_index" in Specs):
        outFile = File(outname, header=infile.header, mode='w')
        outFile.define_new_dimension(name="user_return_index", data_type=5, description="User combined return attr.")

        for dimension in infile.point_format:
            dat = infile.reader.get_dimension(dimension.name)
            outFile.writer.set_dimension(dimension.name, dat)

        USER_ReturnIndex = infile.num_returns * 10
        USER_ReturnIndex = USER_ReturnIndex + infile.return_num

        outFile.user_return_index = USER_ReturnIndex
        return outFile
    else:
        USER_ReturnIndex = infile.num_returns * 10
        USER_ReturnIndex = USER_ReturnIndex + infile.return_num

        infile.user_return_index = USER_ReturnIndex


def rh_cluster(infile, min_points=1, max_points=sys.maxsize, tolerance=1.0):
    """
    Compute cluster ID on a las file. The dimension must be created before running the script.

    :param infile: las file on which to get the cluster id.
    :param outname: name of the las file which will be outputed
    :param min_points: minimum number of points for which we are going to consider a cluster
    :param max_points: maximum number of points to consider the cluster
    :param tolerance: maximum Euclidean distance for a point to be added to the cluster.
    :return: las object
    """

    from sklearn.cluster import DBSCAN

    points = np.stack((infile.x, infile.y, infile.z), axis=1)

    clustering = DBSCAN(eps=tolerance, min_samples=min_points).fit(points)

    labels = clustering.labels_

    infile.cluster_id = labels + 1


def rh_pdal_cluster(inname, outname):
    """
    Performs pdal cluster.
    :param inname: name of the laspy object to read
    :param outname: name of the laspy object to write
    :return:

    """
    command = 'pdal translate {} {} cluster ' \
              '--filters.cluster.tolerance=2.0 ' \
              '--filters.cluster.min_points=1 ' \
              '--writers.las.extra_dims="all"'

    command = command.format(inname, outname)
    os.system(command)


def rh_cluster_median_return(infile, outname):
    """
    Make dimension cluster_id_median and cluter_id_mean.

    :param infile: las file object from which to compute
    :param outname: name file of the output las file
    :return: laspy object
    """

    outFile = File(outname, header=infile.header, mode='w')

    Specs = [spec.name for spec in infile.point_format]

    if not ("cluster_id_median" in Specs):
        outFile.define_new_dimension(name="cluster_id_median", data_type=9, description="Clustering id median")
        ClusterID_Median = np.zeros(len(infile.x))

        for dimension in infile.point_format:
            dat = infile.reader.get_dimension(dimension.name)
            outFile.writer.set_dimension(dimension.name, dat)
    else:
        ClusterID_Median = infile.cluster_id_median

    if not ("cluster_id_mean" in Specs):
        outFile.define_new_dimension(name="cluster_id_median", data_type=9, description="Clustering id mean")
        ClusterID_Mean = np.zeros(len(infile.x))

        for dimension in infile.point_format:
            dat = infile.reader.get_dimension(dimension.name)
            outFile.writer.set_dimension(dimension.name, dat)

    else:
        ClusterID_Mean = infile.cluster_id_mean

    return_number = infile.return_num * 10
    return_number = infile.num_returns + return_number
    cluster_id = infile.cluster_id

    for i in np.unique(np.sort(cluster_id)):
        Mask = cluster_id == i
        Mask_ReturnNumber = return_number[Mask]

        Median = np.median(Mask_ReturnNumber)
        Mean = np.mean(Mask_ReturnNumber)

        ClusterID_Median[Mask] = Median
        ClusterID_Mean[Mask] = Mean

    outFile.cluster_id_median = ClusterID_Median
    outFile.cluster_id_mean = ClusterID_Mean

    return outFile


def rh_cluster_id_count_max(infile):
    """
    Count number in each cluster.

    :param infile: las file on wchi to make dimension
    :param outname: file name for the output file
    :return: laspy object
    """

    ClusterIDCountMax = np.zeros(len(infile.x))

    Specs = [spec.name for spec in infile.point_format]

    if "cluster_id_count_max" in Specs:
        ClusterIDCountMax = infile.cluster_id_count_max

    ClusterID = infile.cluster_id

    for i in np.unique(np.sort(ClusterID)):
        Mask = ClusterID == i

        number_of_points = np.sum(Mask)

        ClusterIDCountMax[Mask] = number_of_points

    ClusterIDCountMax = np.clip(ClusterIDCountMax, 0, 10000)

    infile.cluster_id = ClusterID
    infile.cluster_id_count_max = ClusterIDCountMax


def rh_curvature(infile, multiplier=100):
    """
    Multiply the curvature by some number.

    :param infile: laspy object with curvature dimension
    :param multiplier: number which the curvature will be multiplied by.
    :return:
    """

    Curvature = infile.curvature

    Curvature *= multiplier

    infile.curvature = Curvature


def rh_curvature_change_median(infile, clip=16, classification=15, n_neigh=251):
    """

    :param infile:
    :param clip:
    :param classification:
    :param n_neigh:
    :return:
    """

    cls = infile.classification
    x_array = infile.x
    y_array = infile.y
    z_array = infile.x

    class10 = (cls == classification)

    # XYZ dataset
    coords = np.vstack((x_array, y_array, z_array))

    # XYZ dataset with all classification 15
    coords_flight = np.vstack((x_array[class10], y_array[class10], z_array[class10]))

    if len(coords_flight[0]) != 0:
        # n_neighbors is the number of neighbors of a point, I can't take everything within 1m
        # but I can take a lot of them for now 125
        if len(coords_flight[0]) < 251:
            n_neigh = len(coords_flight[0])
        else:
            n_neigh = 251

        nhbrs = NearestNeighbors(n_neighbors=n_neigh, algorithm="kd_tree").fit(np.transpose(coords_flight))
        distances, indices = nhbrs.kneighbors(np.transpose(coords))

        Curvature = infile.curvature

        Curvature_array = Curvature[indices].astype(float)
        Curvature_array[distances >= clip] = np.nan
        Curvature_avg = np.nanmedian(Curvature_array, axis=1)

        # Apply the median Curvature
        Curvature = Curvature_avg.astype('double')

        # Get the changed Curvature to output
        infile.curvature = Curvature
