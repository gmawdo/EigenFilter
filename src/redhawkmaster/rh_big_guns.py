import csv
import math
import os
import pandas as pd
import numpy as np
from laspy.file import File
from matplotlib import path
from scipy import interpolate
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range
from multiprocessing import cpu_count, Pool


def init_equal_time(bigFile, ss, ee):
    """
    Function to initialize few global arguments for each process
    of the multiprocessing for the tiling with equal gps times.
    You can't use it for anything else!

    :param bigFile: is the input las file in function for tiling
    :type bigFile: laspy object
    :param ss: start gps time for the tiling. 'start' in the function tile_equal time
    :type: ss: float array
    :param ee: end gps time for the tiling. 'end' in the function tile_equal time
    :type ee: float array
    """
    global infile, start, end
    infile = bigFile
    start = ss
    end = ee


def tile_equal_time(i):
    """
    Function that is doing the tiling based on equal gps time.
    You can't use it for anything else!

    :param i: number of the tile.
    :type i: int
    :return: writes a tile to the system
    """
    mask = las_range(infile.gps_time, start[i], end[i])

    outFile = rh_io.las_output(infile.filename.split('/')[-1].split('.las')[0]+"_Tile"+str(i).zfill(3)+".las",
                               infile, mask)
    outFile.close()


def rh_tiling_gps_equal_time(filename, no_tiles=10):
    """
    Starting the multiprocessing pool of threads that are going to
    split the big tile.

    :param filename: File name which needs to be tiled.
    :type filename: string
    :param no_tiles: Number of tiles in which the file will be tiled
    :type no_tiles: int
    """

    # Input the big tile
    bigFile = rh_io.las_input(filename, mode="r")

    # Get the gps time array of that tile
    gps_time = bigFile.gps_time

    # Compute the min and max
    max_gps = max(gps_time)
    min_gps = min(gps_time)

    # Compute the step needed for those number of tiles
    step = ((max_gps - min_gps) / no_tiles)

    # Number of cpu's by 2 is the pool size
    pool_size = cpu_count() * 2
    g_time = min_gps
    start_arr = []
    end_arr = []

    # Get the start and end array for each tile
    for i in range(no_tiles):
        start_arr.append(g_time)
        end_arr.append(g_time + step)
        g_time = g_time + step

    # Init the pool of processes
    pool = Pool(processes=pool_size, initializer=init_equal_time, initargs=(bigFile, start_arr, end_arr))
    # Map the processes from the pool
    pool.map(tile_equal_time, range(no_tiles))
    pool.close()


def init_equal_filesize(bigFile, ss):
    """
    Function to initialize few global arguments for each process
    of the multiprocessing for the tiling with equal file sizes.
    You can't use it for anything else!

    :param bigFile: is the input las file in function for tiling
    :type bigFile: laspy object
    :param ss: digits on which to tile upon on
    :type ss: int array
    """
    global infile, digits
    infile = bigFile
    digits = ss


def tile_equal_filesize(item):
    """
    Function that is doing the tiling based on equal file sizes.
    You can't use it for anything else!

    :param item: number of the tile.
    :type item: int
    :return: writes a tile to the system
    """
    mask = (digits == item)

    outFile = rh_io.las_output(infile.filename.split('/')[-1].split('.las')[0]+"_Tile"+str(item).zfill(3)+".las",
                               infile, mask)
    outFile.close()

 
def rh_tiling_gps_equal_filesize(filename, no_tiles=10):
    """
    Starting the multiprocessing pool of threads that are going to
    split the big tile.

    :param filename: File name which needs to be tiled.
    :type filename: string
    :param no_tiles: Number of tiles in which the file will be tiled
    :type no_tiles: int
    """
    # Input the big file
    bigFile = rh_io.las_input(filename, mode="r")

    # Extract the gps time
    gps_time = bigFile.gps_time

    # Sam's magic (please edit the file with explanation)
    q_values = np.linspace(0, 1, no_tiles + 1)[1:]
    time_quantiles = np.array(list(np.quantile(gps_time, q=a) for a in q_values))
    digits = np.digitize(gps_time, time_quantiles)

    digits[digits == no_tiles] = no_tiles - 1

    # Same as previous
    pool_size = cpu_count() * 2

    pool = Pool(processes=pool_size, initializer=init_equal_filesize, initargs=(bigFile, digits))
    pool.map(tile_equal_filesize, range(no_tiles))
    pool.close()


def rh_extract_ground(inname, outname, slope=0.1, cut=0.0, window=18, cell=1.0, scalar=0.5, threshold=0.5):
    """
    Extraction of ground points. It is making the command
    to run the pdal ground app.

    :param inname: File name on which we need to extract the ground points.
    :type inname: string
    :param outname: File name which will have the ground points classified
    :type outname: string
    :param slope: ground param
    :param cut: ground param
    :param window: ground param
    :param cell: ground param
    :param scalar: ground param
    :param threshold: ground param
    """

    # Example json which we don't need but
    # might use later if we have pdal python library
    json = '{"pipeline": ' \
           '[{"count": "18446744073709551615", ' \
           '"type": "readers.las", ' \
           '"compression": "EITHER", ' \
           '"filename": {}}, ' \
           '{slope": {},' \
           '"cut": {},' \
           '"window": {},' \
           '"cell": {},' \
           '"scalar": {},' \
           '"threshold": {},' \
           '"type": "filters.smrf"' \
           '},' \
           '{' \
           '"extra_dims": "all",' \
           '"type": "writers.las",' \
           '"filename": {}}' \
           '}' \
           ']' \
           '}'

    # The ground command v2 with translate
    ground_command_v2 = "pdal translate " \
                        "--writers.las.extra_dims=all {} {} smrf" \
                        " --filters.smrf.slope={} " \
                        "--filters.smrf.cut={} " \
                        "--filters.smrf.window={} " \
                        "--filters.smrf.cell={} " \
                        "--filters.smrf.scalar={} " \
                        "--filters.smrf.threshold={} "

    # The ground command with pdal ground app
    ground_command = "pdal ground --slope 0.1 --max_window_size 18 --cell_size 0.5 --initial_distance 2.0 -i {} -o {}"

    command_v2 = ground_command_v2.format(inname, outname, slope, cut, window, cell, scalar, threshold)
    command = ground_command.format(inname, outname)
    # print(command)
    os.system(command_v2)


def rh_hag(inname, outname):
    """
    Compute hag (height above ground) dimension on a las file.

    :param inname: File name on which we get the hag attribute
    :type inname: string
    :param outname: File name which will have the hag attribute
    :type outname: string
    :return string output file name
    """
    outname = outname.split('.')[0] + "_hag.las"
    hag_command = 'pdal translate {} {} hag --writers.las.extra_dims="all"'

    command = hag_command.format(inname, outname)
    os.system(command)
    return outname


def rh_hag_smooth(infile, point_id_mask=np.array([])):
    """
    Compute smoother hag (height above ground) dimension on a las file.

    :param infile: laspy object which already has extra dimension heightaboveground
    :type infile: lapsy object
    :param point_id_mask: point id mask
    :type point_id_mask:  int array
    :return array with smooth hag values
    """

    # Get the separated classification and coords
    cls = infile.classification[point_id_mask]
    x_array = infile.x[point_id_mask]
    y_array = infile.y[point_id_mask]
    z_array = infile.z[point_id_mask]

    # Get everything that is ground
    class02 = (cls == 2)

    # Compute median on the elevation of the ground
    df = pd.DataFrame({'X': np.fix(x_array[class02]), 'Y': np.fix(y_array[class02]), 'Z': z_array[class02]})
    grouped = df.groupby(['X', 'Y']).agg({'Z': np.median}).reset_index()

    resid = grouped.values

    # Maybe don't need the coords but still
    xyz_txt = np.vstack((resid[:, 0], resid[:, 1], resid[:, 2]))

    np.savetxt('data.txt', resid, fmt='%s', delimiter=" ")

    # Get everything that is not ground
    class_not02 = (cls != 2)
    x_not02 = x_array[class_not02]
    y_not02 = y_array[class_not02]

    # Interpolate the median values
    f = interpolate.SmoothBivariateSpline(resid[:, 0], resid[:, 1], resid[:, 2], s=len(x_array) * 4)

    # Calculate the new hag Z
    znew = f(x_array, y_array, grid=False)

    # Apply to the new hag
    infile.heightaboveground[point_id_mask] = z_array - znew

    return z_array - znew


def hough_3d(infile, dist=6.4, polygonHeightOuter=1, polygonWidthOuter=0.35,
             polygonHeightInner=1, polygonWidthInner=0.10, m=0.6, pnt_tile=10,
             ext=5, angle_line=65, diff=50, in_pnt=1500, min_Z=-10, max_Z=26,
             classification=14, lOuter=0.3, lClass=0.5):
    """
    With help of hough3dlines we compute lines on a dataset which is divided into small squares.
    File is required to have HeightAboveGround dimension.

    :param infile: file on which we need to find conductor
    :param dist: size of the square tiles
    :param polygonHeightOuter: height of the outer polygon
    :param polygonWidthOuter: width of the outer polygon
    :param polygonHeightInner: height of the inner polygon
    :param polygonWidthInner: width of the inner polygon
    :param m: small extension for the two points of the line
    :param pnt_tile: threshold for points in a small tiles in the dataset
    :param ext: extensions for the polygons
    :param angle_line: angle for the line
    :param diff: difference between the inner and outer polygon threshold
    :param in_pnt: points in the line threshold
    :param min_Z: min elevation of where points for the conductors start
    :param max_Z: max elevation of where points for condcutor end
    :param classification: how the conductor will be classified
    :param lOuter: real classification polygon width
    :param lClass: user classification polygon width
    :return: classed file
    """

    # path of the exe for the hough3dlines
    path_h3d = os.getcwd()+'/redhawkmaster/'

    # Pulling out X,Y,HeightAboveGround, Classification
    cls = infile.Classification

    X_orig = infile.x
    Y_orig = infile.y
    Z_orig = infile.heightaboveground

    # Make a clipping mask
    maskZ = (cls != 2)

    # Apply the mask
    X = X_orig[maskZ]
    Y = Y_orig[maskZ]
    Z = Z_orig[maskZ]

    # If the dataset is empty do not search for conductor
    if len(X) == 0:
        print("Dataset for hough_3d is empty. Exiting!")
        exit()

    # Max and min values for X and Y for 6.4 x 6.4 tiling
    maxX = max(X)
    minX = min(X)

    maxY = max(Y)
    minY = min(Y) + dist/2

    # The helping directories for storing the output of the hough3dlines script
    text_dir = "./TXT"
    output_dir = "./OUTPUT"

    # If they don't exist, create them
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tile number for hough
    tile = 0

    mask = []
    resultLines = []

    for i in np.arange(minX, maxX + dist, dist):
        for j in np.arange(minY, maxY + dist, dist):
            # disti is the X point of the tile and distj is the Y point of the tile
            disti = round(i, 2) + dist
            distj = round(j, 2) + dist

            # Making a mask from the tle points
            maskPart_of_Tile = (X_orig > i) & (X_orig < disti) & (Y_orig > j) & (Y_orig < distj) & (Z_orig > min_Z) & (
                        Z_orig < max_Z) & (cls != 2)

            # Apply the mask
            X_txt = X_orig[maskPart_of_Tile]
            Y_txt = Y_orig[maskPart_of_Tile]
            Z_txt = Z_orig[maskPart_of_Tile]

            # Make (x,y,z) dataset for computing the hough lines
            XYZ_TXT = zip(X_txt, Y_txt, Z_txt)

            # If we have less then some number of points in tile, we are not computing houghlines on that file
            csv.register_dialect('myDialect', delimiter=',', quoting=csv.QUOTE_NONE)

            if len(X_txt) > pnt_tile:
                tile = tile + 1

                # Writing the (x,y,z) dataset to a tile with unique tile number in TXT folder
                myFile = open(text_dir + '/PartOfTile' + str("{:0>3d}".format(tile)) + '_LAS.txt', 'w')

                with myFile:
                    writer = csv.writer(myFile, dialect='myDialect')
                    for row in XYZ_TXT:
                        writer.writerow(row)

                # Call the script hough3dlines and out put the hough lines in a file
                # with unique file name in the OUTPUT folder
                os.system(path_h3d+"hough3dlines " + text_dir + "/PartOfTile" + str('{:0>3d}'.format(
                    tile)) + "_LAS.txt -gnuplot -nlines 9 -minvotes 10 -raw -o " + output_dir + "/output_" + str(
                    "{:0>3d}".format(tile)) + ".txt | gnuplot -persist")

                # Reading the hough lines
                with open(output_dir + '/output_' + str("{:0>3d}".format(tile)) + '.txt', 'r') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=' ')
                    data = [r for r in csv_reader]

                    # If we have lines
                    if len(data) != 0:

                        # TO-DO: First two lines from the output of houghlines are unknown
                        data.pop(0)
                        data.pop(0)

                        for outputLine in data:
                            # Ax, Ay, Az referent point of the line
                            Ax = float(outputLine[0])
                            Ay = float(outputLine[1])
                            Az = float(outputLine[2])

                            # Bx, By, Bz direction of the line
                            Bx = float(outputLine[3])
                            By = float(outputLine[4])
                            Bz = float(outputLine[5])

                            # Small extension and creaton of two points for the polygon extension
                            X1 = Ax + (m * Bx)
                            Y1 = Ay + (m * By)
                            Z1 = Az + (m * Bz)

                            X2 = Ax - (m * Bx)
                            Y2 = Ay - (m * By)
                            Z2 = Az - (m * Bz)

                            # Computing the angle
                            angle = (math.acos(abs(Bz) / math.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)) * 180) / math.pi

                            if angle > angle_line:
                                point = [X1, Y1, Z1, X2, Y2, Z2]

                                # We save the orig values of the two points
                                x1_orig = X1
                                x2_orig = X2

                                y1_orig = Y1
                                y2_orig = Y2

                                z1_orig = Z1
                                z2_orig = Z2

                                # Outer-Inner polygon method
                                # First we create two 2D polygons, one in another
                                # After that we clip by the height which will get us 3D polygons
                                # If we have points in the difference between Outer and Inner polygon
                                # we check how much points, if the number is small then we have a line

                                # Distance between two points
                                d = math.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)

                                # Extension
                                e = ext

                                # Point with extension
                                x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
                                y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
                                x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
                                y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

                                # OUTER POLYGON
                                # How wide is the polygon
                                lOuter = polygonWidthOuter

                                # Distance between extended points
                                DOuter = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                                # XY of the first point of four of the 2D polygon
                                firstX = x1 + (lOuter * (y2 - y1) / DOuter)
                                firstY = y1 + (lOuter * (x1 - x2) / DOuter)

                                # XY of the second point
                                secondX = x1 + (lOuter * (y1 - y2) / DOuter)
                                secondY = y1 + (lOuter * (x2 - x1) / DOuter)

                                # XY of the third point
                                thirdX = x2 + (lOuter * (y2 - y1) / DOuter)
                                thirdY = y2 + (lOuter * (x1 - x2) / DOuter)

                                # XY of the fourth point
                                fourthX = x2 + (lOuter * (y1 - y2) / DOuter)
                                fourthY = y2 + (lOuter * (x2 - x1) / DOuter)

                                # coords of the polygon
                                coordsOuter = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY),
                                               (thirdX, thirdY)]
                                # We create a path using the four points
                                polyOuter = path.Path(coordsOuter)

                                # Check if we have some in that path without limiting the Z
                                inOuter = polyOuter.contains_points(
                                    np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

                                # Now we check that we are not out of the tile coords
                                # and also limiting the Z from the original value of the
                                # z from the line + some polygon height
                                rowOuter = (X_orig > i) & (X_orig < disti) & (Y_orig > j) & (Y_orig < distj) & (
                                        Z_orig > (z1_orig - polygonHeightOuter)) & (
                                                   Z_orig < (z2_orig + polygonHeightOuter))

                                # We get all the points in a mask for the outer polygon
                                indxOuter = np.logical_and(inOuter, rowOuter)

                                # INNER POLYGON
                                # How wide is the inner polygon
                                lInner = polygonWidthInner

                                # Distance between extended points for the Inner polygon
                                DInner = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                                # XY of the first point for the Inner polygon
                                firstX = x1 + (lInner * (y2 - y1) / DInner)
                                firstY = y1 + (lInner * (x1 - x2) / DInner)

                                # XY of the second point
                                secondX = x1 + (lInner * (y1 - y2) / DInner)
                                secondY = y1 + (lInner * (x2 - x1) / DInner)

                                # XY of the third point
                                thirdX = x2 + (lInner * (y2 - y1) / DInner)
                                thirdY = y2 + (lInner * (x1 - x2) / DInner)

                                # XY of the fourth point
                                fourthX = x2 + (lInner * (y1 - y2) / DInner)
                                fourthY = y2 + (lInner * (x2 - x1) / DInner)

                                # coords of the 2D polygon
                                coordsInner = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY),
                                               (thirdX, thirdY)]
                                # Create a path using those four points
                                polyInner = path.Path(coordsInner)

                                # check if we have some points in that path without limiting the z
                                inInner = polyInner.contains_points(
                                    np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

                                # Same as for outer
                                rowInner = (X_orig > i) & (X_orig < disti) & (Y_orig > j) & (Y_orig < distj) & (
                                        Z_orig > (z1_orig - polygonHeightInner)) & (
                                                   Z_orig < (z2_orig + polygonHeightInner))

                                # We get all the points in a mas for the outer polygon
                                indxInner = np.logical_and(inInner, rowInner)

                                # Compute the difference between Outer and Inner polygon points
                                indx = np.logical_xor(indxInner, indxOuter).ravel().nonzero()

                                # If that difference is less then 50 points we have a conductor
                                if len(indx[0]) < diff:
                                    if len(indxInner) != 0 & np.sum(indxInner) < in_pnt:
                                        mask.append(indxInner.ravel().nonzero())
                                        resultLines.append(outputLine)

    # For the real classification
    wholeResult = []
    # For User classification
    userResult = []

    # Openning result file hough lines for vegetation selection
    f = open('./hough_result_lines_' + str(1).zfill(2) + '.txt', 'a')
    for line in resultLines:
        Ax = float(line[0])
        Ay = float(line[1])
        Az = float(line[2])

        Bx = float(line[3])
        By = float(line[4])
        Bz = float(line[5])

        f.write("%f %f %f %f %f %f\n" % (Ax, Ay, Az, Bx, By, Bz))

        X1 = Ax + (m * Bx)
        Y1 = Ay + (m * By)
        Z1 = Az + (m * Bz)

        X2 = Ax - (m * Bx)
        Y2 = Ay - (m * By)
        Z2 = Az - (m * Bz)

        x1_orig = X1
        x2_orig = X2
        y1_orig = Y1
        y2_orig = Y2
        z1_orig = Z1
        z2_orig = Z2

        d = math.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)

        e = 5
        x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
        y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
        x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
        y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

        # OUTER POLYGON

        DOuter = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        firstX = x1 + (lOuter * (y2 - y1) / DOuter)
        firstY = y1 + (lOuter * (x1 - x2) / DOuter)

        secondX = x1 + (lOuter * (y1 - y2) / DOuter)
        secondY = y1 + (lOuter * (x2 - x1) / DOuter)

        thirdX = x2 + (lOuter * (y2 - y1) / DOuter)
        thirdY = y2 + (lOuter * (x1 - x2) / DOuter)

        fourthX = x2 + (lOuter * (y1 - y2) / DOuter)
        fourthY = y2 + (lOuter * (x2 - x1) / DOuter)

        coordsOuter = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY), (thirdX, thirdY)]
        polyOuter = path.Path(coordsOuter)

        inWhole = polyOuter.contains_points(
            np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

        rowWhole = (Z_orig > (z1_orig - 0.5)) & (Z_orig < (z2_orig + 0.5)) & (cls != 2)

        e = ext
        x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
        y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
        x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
        y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

        # USER CLASSIFICATION POLYGON

        DClass = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        firstX = x1 + (lClass * (y2 - y1) / DClass)
        firstY = y1 + (lClass * (x1 - x2) / DClass)

        secondX = x1 + (lClass * (y1 - y2) / DClass)
        secondY = y1 + (lClass * (x2 - x1) / DClass)

        thirdX = x2 + (lClass * (y2 - y1) / DClass)
        thirdY = y2 + (lClass * (x1 - x2) / DClass)

        fourthX = x2 + (lClass * (y1 - y2) / DClass)
        fourthY = y2 + (lClass * (x2 - x1) / DClass)

        coordsClass = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY), (thirdX, thirdY)]
        polyClass = path.Path(coordsClass)

        inClass = polyClass.contains_points(
            np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

        rowUser = (Z_orig > min_Z) & (Z_orig < max_Z) & (cls != 2)

        # At the end we are appending only indexes of the points that we need to extract
        # With this way it is easier to make a full mask for the classification of the
        # conductor.
        wholeResult.append(np.logical_and(inWhole, rowWhole).ravel().nonzero())
        userResult.append(np.logical_and(inClass, rowUser).ravel().nonzero())

    f.close()

    if len(wholeResult) != 0:
        maskWhole = np.concatenate(wholeResult, axis=1)[0]
        maskUser_CLS = np.concatenate(userResult, axis=1)[0]

        # Array of all false
        # testing:maskFalse = np.zeros((1,len(X_orig)),dtype=bool)[0]

        # Regular classification
        maskResult = np.zeros((1, len(X_orig)), dtype=bool)[0]

        # User classification
        maskUser = np.zeros((1, len(X_orig)), dtype=bool)[0]

        # On selected indexes make the value true of the mask
        # testing: maskFalse[np.array(maskC)] = True

        maskResult[np.array(maskWhole)] = True

        maskUser[np.array(maskUser_CLS)] = True

        cls[maskResult] = classification
        # user_cls[maskUser] = 14

        # Used for testing
        # outs['Mask'] = maskFalse

        infile.Classification = cls
        print("Found " + str(len(wholeResult)) + " conductor lines!!")
        # Delete Output and TXT dirs
        os.system("rm -r ./OUTPUT/")
        os.system("rm -r ./TXT/")
        # outs['USER_CLASS'] = user_cls
    else:
        print("There is no cable found")


def apply_hough(infile, tile_no='01', polygonHeightOuter=1, polygonWidthOuter=0.35,
                polygonHeightInner=1, polygonWidthInner=0.10, m=0.6,
                ext=5, diff=1000, in_pnt=1500, classification=14):
    """
    Apply hough result lines on 090 dataset.

    :param infile: file on which we need to find conductor
    :param tile_no: string of tile nnumber
    :param polygonHeightOuter: height of the outer polygon
    :param polygonWidthOuter: width of the outer polygon
    :param polygonHeightInner: height of the inner polygon
    :param polygonWidthInner: width of the inner polygon
    :param m: small extension for the two points of the line
    :param ext: extensions for the polygons
    :param diff: difference between the inner and outer polygon threshold
    :param in_pnt: points in the line threshold
    :param classification: how the conductor will be classified
    :return: classed file
    """

    cls = infile.classification

    X_orig = infile.x
    Y_orig = infile.y
    Z_orig = infile.heightaboveground

    mask = []

    f = open('./hough_applied_lines_' + tile_no + '.txt', 'w')

    try:
        fb = open('./hough_result_lines_' + tile_no + '.txt', 'r')
        fb.close()
    except IOError as e:
        print("There is not any hough lines, Returning true.")
        exit()

    with open('./hough_result_lines_' + tile_no + '.txt', 'r') as result_file:
        csv_reader = csv.reader(result_file, delimiter=' ')
        data = [r for r in csv_reader]

        for line in data:

            Ax = float(line[0])
            Ay = float(line[1])
            Az = float(line[2])

            Bx = float(line[3])
            By = float(line[4])
            Bz = float(line[5])

            x1_orig = Ax + (m * Bx)
            y1_orig = Ay + (m * By)
            z1_orig = Az + (m * Bz)

            x2_orig = Ax - (m * Bx)
            y2_orig = Ay - (m * By)
            z2_orig = Az - (m * Bz)

            d = math.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)

            # Extension by 10m
            e = ext
            x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
            y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
            x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
            y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

            # OUTER POLYGON
            lOuter = polygonWidthOuter

            DOuter = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            firstX = x1 + (lOuter * (y2 - y1) / DOuter)
            firstY = y1 + (lOuter * (x1 - x2) / DOuter)

            secondX = x1 + (lOuter * (y1 - y2) / DOuter)
            secondY = y1 + (lOuter * (x2 - x1) / DOuter)
            thirdX = x2 + (lOuter * (y2 - y1) / DOuter)
            thirdY = y2 + (lOuter * (x1 - x2) / DOuter)

            fourthX = x2 + (lOuter * (y1 - y2) / DOuter)
            fourthY = y2 + (lOuter * (x2 - x1) / DOuter)

            coordsOuter = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY), (thirdX, thirdY)]
            polyOuter = path.Path(coordsOuter)

            inOuter = polyOuter.contains_points(
                np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

            rowOuter = (Z_orig > (z1_orig - polygonHeightOuter)) & (Z_orig < (z2_orig + polygonHeightOuter))

            indxOuter = np.logical_and(inOuter, rowOuter)

            # INNER POLYGON
            lInner = polygonWidthInner

            DInner = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            firstX = x1 + (lInner * (y2 - y1) / DInner)
            firstY = y1 + (lInner * (x1 - x2) / DInner)

            secondX = x1 + (lInner * (y1 - y2) / DInner)
            secondY = y1 + (lInner * (x2 - x1) / DInner)

            thirdX = x2 + (lInner * (y2 - y1) / DInner)
            thirdY = y2 + (lInner * (x1 - x2) / DInner)

            fourthX = x2 + (lInner * (y1 - y2) / DInner)
            fourthY = y2 + (lInner * (x2 - x1) / DInner)

            coordsInner = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY), (thirdX, thirdY)]
            polyInner = path.Path(coordsInner)

            inInner = polyInner.contains_points(
                np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

            rowInner = (Z_orig > (z1_orig - polygonHeightInner)) & (Z_orig < (z2_orig + polygonHeightInner))

            indxInner = np.logical_and(inInner, rowInner)

            indx = np.logical_xor(indxInner, indxOuter).ravel().nonzero()

            # mask.append(indx)
            if len(indx[0]) < diff:
                if len(indxInner) != 0 & np.sum(indxInner) < in_pnt:
                    f.write(str(Ax)+' '+str(Ay)+' '+str(Az)+' '+str(Bx)+' '+str(By)+' '+str(Bz)+'\n')
                    mask.append(indxInner.ravel().nonzero())
                    # FINISH method

    f.close()

    if len(mask) != 0:

        maskC = np.concatenate(mask, axis=1)[0]

        maskFalse = np.zeros((1, len(X_orig)), dtype=bool)[0]

        maskFalse[np.array(maskC)] = True

        cls[maskFalse] = classification

        infile.classification = cls

        # outs['Mask'] = maskFalse
    else:
        print("We don't find lines in hough result lines.")


def corridor(infile, tile_no='01', ext=10, m=0.6, min_Z=-10, max_Z=22, lOuter=0.3):
    """
    Make a corridor around hough applied lines.

    :param infile:
    :param tile_no:
    :param ext:
    :param m:
    :param min_Z:
    :param max_Z:
    :param lOuter:
    :return:
    """

    X_orig = infile.x
    Y_orig = infile.y
    Z_orig = infile.heightaboveground

    mask = []

    with open('./hough_applied_lines_' + tile_no + '.txt', 'r') as result_file:
        csv_reader = csv.reader(result_file, delimiter=' ')
        data = [r for r in csv_reader]

        for line in data:
            Ax = float(line[0])
            Ay = float(line[1])
            Az = float(line[2])

            Bx = float(line[3])
            By = float(line[4])
            Bz = float(line[5])

            x1_orig = Ax + (m * Bx)
            y1_orig = Ay + (m * By)

            x2_orig = Ax - (m * Bx)
            y2_orig = Ay - (m * By)

            d = math.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)

            e = ext

            x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
            y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
            x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
            y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

            DOuter = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            firstX = x1 + (lOuter * (y2 - y1) / DOuter)
            firstY = y1 + (lOuter * (x1 - x2) / DOuter)

            secondX = x1 + (lOuter * (y1 - y2) / DOuter)
            secondY = y1 + (lOuter * (x2 - x1) / DOuter)
            thirdX = x2 + (lOuter * (y2 - y1) / DOuter)
            thirdY = y2 + (lOuter * (x1 - x2) / DOuter)

            fourthX = x2 + (lOuter * (y1 - y2) / DOuter)
            fourthY = y2 + (lOuter * (x2 - x1) / DOuter)

            coordsOuter = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY), (thirdX, thirdY)]
            polyOuter = path.Path(coordsOuter)

            inOuter = polyOuter.contains_points(
                np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

            rowOuter = (Z_orig > min_Z) & (Z_orig < max_Z)

            indxOuter = np.logical_and(inOuter, rowOuter)

            if len(indxOuter) != 0:
                mask.append(indxOuter.ravel().nonzero())

    if len(mask) != 0:

        maskC = np.concatenate(mask, axis=1)[0]
        maskFalse = np.zeros((1, len(X_orig)), dtype=bool)[0]

        maskFalse[np.array(maskC)] = True

        return maskC

        # outs['Mask'] = maskFalse
    else:
        maskFalse = range(1, len(X_orig))

        print("There aren't cables classified to make corridor.")
        return maskFalse


def pylon_extract(infile, min_mask_Z=4, max_mask_Z=12, dist=6.4, m=0.6, diff=20,
                  angle_line=5, ext=0.5, lOuter=0.5, min_pylon_Z=0, max_pylon_Z=12,
                  classification=13):
    """
    Extract pylons from a dataset.

    :param infile:
    :param min_mask_Z:
    :param max_mask_Z:
    :param dist:
    :param m:
    :param diff:
    :param angle_line:
    :param ext:
    :param lOuter:
    :param min_pylon_Z:
    :param max_pylon_Z:
    :param classification:
    :return:
    """

    # path of the exe for the hough3dlines
    path_h3d = os.getcwd() + '/redhawkmaster/'

    cls = infile.classification

    X_orig = infile.x
    Y_orig = infile.y
    Z_orig = infile.heightaboveground

    # Make a clipping mask
    maskZ = (Z_orig > min_mask_Z) & (Z_orig < max_mask_Z)

    # Apply the mask
    X = X_orig[maskZ]
    Y = Y_orig[maskZ]
    Z = Z_orig[maskZ]

    if len(X) == 0:
        print("There is no dataset to classify pylons. Exiting")
        exit()

    maxX = max(X)
    minX = min(X)

    minY = min(Y)
    maxY = max(Y)

    text_dir = './Pylon_TXT_'
    output_dir = './Pylon_OUTPUT_'

    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask = []
    resultLines = []
    tile = 0

    f = open('./hough_pylon_lines_01.txt', 'w')

    for i in np.arange(minX, maxX+dist, dist):
        for j in np.arange(minY, maxY+dist, dist):
            disti = round(i, 2) + dist
            distj = round(j, 2) + dist

            maskPart_of_Tile = (X_orig > i) & (X_orig < disti) & (Y_orig > j) & (Y_orig < distj) & (Z_orig > min_mask_Z) & (
                        Z_orig < max_mask_Z)

            X_txt = X_orig[maskPart_of_Tile]
            Y_txt = Y_orig[maskPart_of_Tile]
            Z_txt = Z_orig[maskPart_of_Tile]
            XYZ_TXT = zip(X_txt, Y_txt, Z_txt)

            csv.register_dialect('myDialect', delimiter=',', quoting=csv.QUOTE_NONE)

            if len(X_txt) > diff:
                tile = tile + 1
                myFile = open(text_dir + '/PartOfTile' + str("{:0>3d}".format(tile)) + '_LAS.txt', 'w')
                with myFile:
                    writer = csv.writer(myFile, dialect='myDialect')
                    for row in XYZ_TXT:
                        writer.writerow(row)

                os.system(path_h3d + "hough3dlines " + text_dir + "/PartOfTile" + str('{:0>3d}'.format(
                    tile)) + "_LAS.txt -gnuplot -nlines 9 -minvotes 10 -raw -o " + output_dir + "/output_" + str(
                    "{:0>3d}".format(tile)) + ".txt | gnuplot -persist")

                with open(output_dir + '/output_' + str("{:0>3d}".format(tile)) + '.txt', 'r') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=' ')
                    data = [r for r in csv_reader]

                    if len(data) != 0:
                        data.pop(0)
                        data.pop(0)

                        for outputLine in data:
                            Ax = float(outputLine[0])
                            Ay = float(outputLine[1])
                            Az = float(outputLine[2])

                            Bx = float(outputLine[3])
                            By = float(outputLine[4])
                            Bz = float(outputLine[5])

                            X1 = Ax + (m * Bx)
                            Y1 = Ay + (m * By)
                            Z1 = Az + (m * Bz)

                            point = [X, Y, Z]

                            X2 = Ax - (m * Bx)
                            Y2 = Ay - (m * By)
                            Z2 = Az - (m * Bz)

                            point2 = [X, Y, Z]

                            angle = (math.acos(abs(Bz) / math.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)) * 180) / math.pi

                            if angle < angle_line:
                                point = [X1, Y1, Z1, X2, Y2, Z2]
                                x1_orig = X1
                                x2_orig = X2
                                y1_orig = Y1
                                y2_orig = Y2
                                z1_orig = Z1
                                z2_orig = Z2

                                d = math.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)

                                # Extension by 1m
                                e = ext

                                x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
                                y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
                                x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
                                y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

                                DOuter = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                firstX = x1 + (lOuter * (y2 - y1) / DOuter)
                                firstY = y1 + (lOuter * (x1 - x2) / DOuter)

                                secondX = x1 + (lOuter * (y1 - y2) / DOuter)
                                secondY = y1 + (lOuter * (x2 - x1) / DOuter)

                                thirdX = x2 + (lOuter * (y2 - y1) / DOuter)
                                thirdY = y2 + (lOuter * (x1 - x2) / DOuter)

                                fourthX = x2 + (lOuter * (y1 - y2) / DOuter)
                                fourthY = y2 + (lOuter * (x2 - x1) / DOuter)

                                coordsOuter = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY),
                                               (thirdX, thirdY)]
                                polyOuter = path.Path(coordsOuter)

                                inOuter = polyOuter.contains_points(
                                    np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

                                rowOuter = (Z_orig > min_pylon_Z) & (Z_orig < max_pylon_Z)

                                # print np.sum(checkClass)

                                indxOuter = np.logical_and(inOuter, rowOuter).ravel().nonzero()

                                # mask.append(indx)

                                if len(indxOuter[0]) != 0:
                                    f.write("%f %f %f %f %f %f\n" % (Ax, Ay, Az, Bx, By, Bz))
                                    mask.append(indxOuter)
                                    resultLines.append(outputLine)

    f.close()
    # print len(mask)

    if len(mask) != 0:
        maskC = np.concatenate(mask, axis=1)[0]

        maskFalse = np.zeros((1, len(X_orig)), dtype=bool)[0]

        maskFalse[np.array(maskC)] = True

        cls[maskFalse] = classification

        # outs['Mask'] = maskFalse
        infile.classification = cls
        os.system("rm -r "+text_dir)
        os.system("rm -r "+output_dir)
    else:
        print('We dont have pylons')


def apply_pylon(infile, tile_no='01', m=0.6, ext=0.5, lOuter=0.5, min_pylon_Z=-25, max_pylon_Z=25,
                min_diff_Z=5.0, max_diff_Z=12, classification=13):
    """
    Apply pylon hough lines on 130 dataset.

    :param infile:
    :param tile_no:
    :param m:
    :param ext:
    :param lOuter:
    :param min_pylon_Z:
    :param max_pylon_Z:
    :param min_diff_Z:
    :param max_diff_Z:
    :param classification:
    :return:
    """
    cls = infile.classification

    X_orig = infile.x
    Y_orig = infile.y
    Z_orig = infile.heightaboveground

    mask = []

    try:
        f = open('./hough_pylon_lines_' + tile_no + '.txt', 'r')
        f.close()
    except IOError as e:
        print("Required hough file doesn't exist. Exiting")
        exit()

    with open('./hough_pylon_lines_' + tile_no + '.txt', 'r') as result_file:
        csv_reader = csv.reader(result_file, delimiter=' ')
        data = [r for r in csv_reader]

        for line in data:
            Ax = float(line[0])
            Ay = float(line[1])
            Az = float(line[2])

            Bx = float(line[3])
            By = float(line[4])
            Bz = float(line[5])

            x1_orig = Ax + (m * Bx)
            y1_orig = Ay + (m * By)
            z1_orig = Az + (m * Bz)

            x2_orig = Ax - (m * Bx)
            y2_orig = Ay - (m * By)
            z2_orig = Az - (m * Bz)

            d = math.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)

            # Extension by 10m
            e = ext

            x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
            y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
            x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
            y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

            DOuter = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            firstX = x1 + (lOuter * (y2 - y1) / DOuter)
            firstY = y1 + (lOuter * (x1 - x2) / DOuter)

            secondX = x1 + (lOuter * (y1 - y2) / DOuter)
            secondY = y1 + (lOuter * (x2 - x1) / DOuter)
            thirdX = x2 + (lOuter * (y2 - y1) / DOuter)
            thirdY = y2 + (lOuter * (x1 - x2) / DOuter)

            fourthX = x2 + (lOuter * (y1 - y2) / DOuter)
            fourthY = y2 + (lOuter * (x2 - x1) / DOuter)

            coordsOuter = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY), (thirdX, thirdY)]
            polyOuter = path.Path(coordsOuter)

            inOuter = polyOuter.contains_points(
                np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

            rowOuter = (Z_orig > min_pylon_Z) & (Z_orig < max_pylon_Z)

            indxOuter = np.logical_and(inOuter, rowOuter).ravel().nonzero()

            cluster = np.logical_and(inOuter, rowOuter)

            diff_z = max(Z_orig[cluster]) - min(Z_orig[cluster])
            # print str(max(Z_orig[cluster])) +" " + str(min(Z_orig[cluster])) + " " + str(diff_z)

            if len(indxOuter[0]) != 0:
                if (diff_z > min_diff_Z) and (diff_z < max_diff_Z):
                    mask.append(indxOuter)

    if len(mask) != 0:
        maskC = np.concatenate(mask, axis=1)[0]

        maskFalse = np.zeros((1, len(X_orig)), dtype=bool)[0]

        maskFalse[np.array(maskC)] = True

        cls[maskFalse] = classification

        infile.classification = cls
        # outs['Mask'] = maskFalse
    else:
        maskC = np.arange(1, len(X_orig))

    return maskC


def classify_vegetation(infile, tile_no='01', m=0.6, polygonWidth=3, polygonHeight=3, ext=7,
                        classification_ground=2, classification_noise=7, classification_flightline=10,
                        classification_pylon=13, classification_conductor=14, classification_vegetation=15):
    """

    :param infile:
    :param tile_no:
    :param m:
    :param polygonWidth:
    :param polygonHeight:
    :param ext:
    :param classification_ground:
    :param classification_noise:
    :param classification_flightline:
    :param classification_pylon:
    :param classification_conductor:
    :param classification_vegetation:
    :return:
    """

    cls = infile.classification
    X_orig = infile.x
    Y_orig = infile.y
    Z_orig = infile.heightaboveground

    mask = []

    with open('./hough_applied_lines_' + tile_no + '.txt', 'r') as result_file:
        csv_reader = csv.reader(result_file, delimiter=' ')

        data = [r for r in csv_reader]

        for line in data:
            Ax = float(line[0])
            Ay = float(line[1])
            Az = float(line[2])

            Bx = float(line[3])
            By = float(line[4])
            Bz = float(line[5])

            x1_orig = Ax + (m * Bx)
            y1_orig = Ay + (m * By)
            z1_orig = Az + (m * Bz)

            x2_orig = Ax - (m * Bx)
            y2_orig = Ay - (m * By)
            z2_orig = Az - (m * Bz)

            d = math.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)

            e = ext
            x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
            y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
            x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
            y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

            # POLYGON
            l = polygonWidth

            D = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            firstX = x1 + (l * (y2 - y1) / D)
            firstY = y1 + (l * (x1 - x2) / D)

            secondX = x1 + (l * (y1 - y2) / D)
            secondY = y1 + (l * (x2 - x1) / D)

            thirdX = x2 + (l * (y2 - y1) / D)
            thirdY = y2 + (l * (x1 - x2) / D)

            fourthX = x2 + (l * (y1 - y2) / D)
            fourthY = y2 + (l * (x2 - x1) / D)

            coords = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY), (thirdX, thirdY)]
            poly = path.Path(coords)

            inPolygon = poly.contains_points(
                np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

            rowPolygon = (Z_orig > (z1_orig - polygonHeight)) & (Z_orig < (z2_orig + polygonHeight)) & \
                         (cls != classification_ground) & (cls != classification_noise) & \
                         (cls != classification_flightline) & (cls != classification_pylon) & \
                         (cls != classification_conductor)

            indx = np.logical_and(inPolygon, rowPolygon)

            conductor = (cls == classification_conductor)

            diff = np.logical_and(indx, conductor).ravel().nonzero()

            indx = indx.ravel().nonzero()

            if len(indx[0]) != 0 and len(diff[0]) == 0:
                mask.append(indx)

    if len(mask) != 0:
        maskC = np.concatenate(mask, axis=1)[0]
        maskFalse = np.zeros((1, len(X_orig)), dtype=bool)[0]

        maskFalse[np.array(maskC)] = True
        cls[np.logical_and(maskFalse, np.logical_not(cls == classification_conductor))] = classification_vegetation

        infile.classification = cls
    else:
        maskC = np.arange(1, len(X_orig))
        # outs['Mask'] = maskFalse

    return maskC


def extract_shape_conductors(infile, shape_path='./', ext=5, lOuter=20):

    X_orig = infile.x
    Y_orig = infile.y

    if len(X_orig) == 0:
        print("The dataset doesn\'t contain lines from the shapefile.Exiting")
        exit()

    minX = min(X_orig)
    maxX = max(X_orig)

    import shapefile
    shp_file_base = shape_path + 'SHAPE/SO80_ohl_11kV'
    sf = shapefile.Reader(shp_file_base)

    shp_file_base_33 = shape_path + '/SHAPE/SO80_ohl_33kV'
    sh_33 = shapefile.Reader(shp_file_base_33)

    shp_file_base_68 = shape_path + '/SHAPE/ST68_ohl_11kV'
    sh_68 = shapefile.Reader(shp_file_base_68)

    shp_file_base_132 = shape_path + 'SHAPE/ST88_ohl_11kV'
    sh_132 = shapefile.Reader(shp_file_base_132)

    mask = []

    combined = list(sf.iterShapes()) + list(sh_33.iterShapes()) + list(sh_132.iterShapes()) + list(sh_68.iterShapes())

    for shape in combined:
        x_lon = np.zeros((len(shape.points), 1))
        y_lat = np.zeros((len(shape.points), 1))
        #        print shape.points
        for ip in range(len(shape.points) - 1):
            firstPoint = shape.points[ip]
            secondPoint = shape.points[ip + 1]

            x1_orig = firstPoint[0]
            x2_orig = secondPoint[0]
            y1_orig = firstPoint[1]
            y2_orig = secondPoint[1]

            if x1_orig < minX or x1_orig > maxX:
                break

            d = math.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)
            if d == 0:
                d = 1

            e = ext
            x1 = x1_orig + ((e / d) * (x1_orig - x2_orig))
            y1 = y1_orig + ((e / d) * (y1_orig - y2_orig))
            x2 = x2_orig + ((e / d) * (x2_orig - x1_orig))
            y2 = y2_orig + ((e / d) * (y2_orig - y1_orig))

            DOuter = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if DOuter == 0:
                DOuter = 1
            firstX = x1 + (lOuter * (y2 - y1) / DOuter)
            firstY = y1 + (lOuter * (x1 - x2) / DOuter)

            secondX = x1 + (lOuter * (y1 - y2) / DOuter)
            secondY = y1 + (lOuter * (x2 - x1) / DOuter)

            thirdX = x2 + (lOuter * (y2 - y1) / DOuter)
            thirdY = y2 + (lOuter * (x1 - x2) / DOuter)

            fourthX = x2 + (lOuter * (y1 - y2) / DOuter)
            fourthY = y2 + (lOuter * (x2 - x1) / DOuter)

            coordsOuter = [(firstX, firstY), (secondX, secondY), (fourthX, fourthY), (thirdX, thirdY)]
            polyOuter = path.Path(coordsOuter)
            inWhole = polyOuter.contains_points(
                np.hstack((X_orig.flatten()[:, np.newaxis], Y_orig.flatten()[:, np.newaxis])))

            if np.sum(inWhole) != 0:
                mask.append(inWhole.ravel().nonzero())

    if len(mask) != 0:
        maskC = np.concatenate(mask, axis=1)[0]

        maskFalse = np.zeros((1, len(X_orig)), dtype=bool)[0]
        maskFalse[np.array(maskC)] = True

    else:
        maskFalse = np.zeros((1, len(X_orig)), dtype=bool)[0]
        maskC = np.arange(1, len(X_orig))
        print("No lines were found. Returning empty dataset")

    return maskC
