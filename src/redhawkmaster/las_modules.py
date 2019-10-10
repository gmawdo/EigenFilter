from laspy.file import File
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
import os


def las_range(dimension, start=(-sys.maxsize-1), end=sys.maxsize, reverse=False, point_id_mask=np.array([])):
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

    inFile = infile

    outFile = File("T000_extradim.las", mode="w", header=inFile.header)

    outFile.define_new_dimension(name=attribute_out, data_type=attr_type, description=attr_descrp)

    for dimension in inFile.point_format:
        dat = inFile.reader.get_dimension(dimension.name)
        outFile.writer.set_dimension(dimension.name, dat)

    outFile.close()
    inFile = File("T000_extradim.las", mode="r")

    outFile1 = File(infile.filename.split('/')[-1].split('.las')[0]+"_"+attribute_out+".las",
                    mode="w", header=inFile.header)

    for dimension in inFile.point_format:
        dat = inFile.reader.get_dimension(dimension.name)
        outFile1.writer.set_dimension(dimension.name, dat)

    in_spec = inFile.reader.get_dimension(attribute_in)
    outFile1.writer.set_dimension(attribute_out, in_spec)
    os.system('rm T000_extradim.las')
    return outFile1


def flightline_point_counter(infile, clip, nh, mask):

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

    cls = infile.Classification
    x_array = infile.x
    y_array = infile.y
    z_array = infile.z

    coords = np.vstack((x_array, y_array, z_array))

    for i in range(0,num_itter):
        class1 = (cls==classif)

        coords_flight = np.vstack((x_array[class1], y_array[class1], z_array[class1]))
        if len(coords_flight[0]) != 0:
            nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(np.transpose(coords_flight))
            distances, indices = nhbrs.kneighbors(np.transpose(coords))

            cls[np.logical_and(distances[:, 0] < clip, np.logical_not(class1))] = classif

    return cls


def rh_clip(infile, clip=100, cls_int=10, point_id_mask=np.array([])):

    classification = infile.Classification[point_id_mask]
    x_array = infile.x[point_id_mask]
    y_array = infile.y[point_id_mask]

    class10 = (classification == cls_int)

    coords = np.vstack((x_array, y_array))

    coords_flight = np.vstack((x_array[class10], y_array[class10]))

    if len(x_array[class10]) == 0:
        print("There is not flightline in this dataset.")
        return 0

    nhbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(np.transpose(coords_flight))
    distances, indices = nhbrs.kneighbors(np.transpose(coords))

    mask = np.logical_and(distances[:, 0] < clip, np.logical_not(class10))

    cls_mask = np.logical_and(np.logical_not(mask), np.logical_not(class10))

    classification[cls_mask] = 7

    cls_tmp = infile.classification
    cls_tmp[point_id_mask] = classification
    infile.classification = cls_tmp

    return classification


def rh_kdistance(infile, output_file_name='', k=10, make_dimension=True):

    k = k + 1

    coords = np.vstack((infile.x, infile.y, infile.z))
    nhbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(np.transpose(coords))
    distances, indices = nhbrs.kneighbors(np.transpose(coords))

    if make_dimension:
        outFile = File(output_file_name, mode="w", header=infile.header)
        outFile.define_new_dimension(name="kdistance", data_type=10, description="KDistance")

        for dimension in infile.point_format:
            dat = infile.reader.get_dimension(dimension.name)
            outFile.writer.set_dimension(dimension.name, dat)

        outFile.writer.set_dimension("kdistance", distances[:, -1])
        return outFile
    else:
        infile.kdistance = distances[:, -1]


def rh_assign(dimension, value, mask):
    cls = dimension
    cls[mask] = value
    return cls
