from laspy.file import File
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
import os


def las_range(dimension, start=(-sys.maxsize-1), end=sys.maxsize, reverse=False, point_id_mask=np.array([])):
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
    mask = (dimension >= start) & (dimension < end)
    if reverse:
        mask = ~mask

    if point_id_mask.size != 0:
        return point_id_mask[mask]
    else:
        return mask


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

    # Open output file with same name as infile just added the dimensions name
    outFile1 = File(infile.filename.split('/')[-1].split('.las')[0]+"_"+attribute_out+".las",
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


def rh_kdistance(infile, output_file_name='', k=10, make_dimension=True):
    """
    Compute kd distance on a file.

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
    coords = np.vstack((infile.x, infile.y, infile.z))

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
        infile.kdistance = distances[:, -1]


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
