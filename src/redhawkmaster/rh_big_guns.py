import os
import pandas as pd
import numpy as np
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
