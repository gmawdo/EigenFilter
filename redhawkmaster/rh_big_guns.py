import os
import pandas as pd
import numpy as np
from scipy import interpolate
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range
from multiprocessing import cpu_count, Pool


def init_equal_time(bigFile, ss, ee):
    global infile, start, end
    infile = bigFile
    start = ss
    end = ee


def tile_equal_time(i):
    mask = las_range(infile.gps_time, start[i], end[i])

    outFile = rh_io.las_output(infile.filename.split('/')[-1].split('.las')[0]+"_Tile"+str(i).zfill(3)+".las",
                               infile, mask)
    outFile.close()


def rh_tiling_gps_equal_time(filename, no_tiles=10):
    bigFile = rh_io.las_input(filename, mode="r")

    gps_time = bigFile.gps_time

    max_gps = max(gps_time)
    min_gps = min(gps_time)

    step = ((max_gps - min_gps) / no_tiles)

    pool_size = cpu_count() * 2
    g_time = min_gps
    start_arr = []
    end_arr = []
    for i in range(no_tiles):
        start_arr.append(g_time)
        end_arr.append(g_time + step)
        g_time = g_time + step

    pool = Pool(processes=pool_size, initializer=init_equal_time, initargs=(bigFile, start_arr, end_arr))
    pool.map(tile_equal_time, range(no_tiles))
    pool.close()


def init_equal_filesize(bigFile, ss):
    global infile, digits
    infile = bigFile
    digits = ss


def tile_equal_filesize(item):
    mask = (digits == item)

    outFile = rh_io.las_output(infile.filename.split('/')[-1].split('.las')[0]+"_Tile"+str(item).zfill(3)+".las",
                               infile, mask)
    outFile.close()


def rh_tiling_gps_equal_filesize(filename, no_tiles=10):
    bigFile = rh_io.las_input(filename, mode="r")

    gps_time = bigFile.gps_time

    q_values = np.linspace(0, 1, no_tiles + 1)[1:]
    time_quantiles = np.array(list(np.quantile(gps_time, q=a) for a in q_values))
    digits = np.digitize(gps_time, time_quantiles)

    digits[digits == no_tiles] = no_tiles - 1

    pool_size = cpu_count() * 2

    pool = Pool(processes=pool_size, initializer=init_equal_filesize, initargs=(bigFile, digits))
    pool.map(tile_equal_filesize, range(no_tiles))
    pool.close()


def rh_extract_ground(inname, outname):

    ground_command = "pdal ground --slope 0.15 --max_window_size 18 --cell_size 0.5 --initial_distance 2.0 -i {} -o {}"

    command = ground_command.format(inname, outname)
    os.system(command)


def rh_hag(inname, outname):

    hag_command = 'pdal translate {} {} hag --writers.las.extra_dims="all"'

    command = hag_command.format(inname, outname)
    os.system(command)


def rh_hag_smooth(infile):

    cls = infile.classification
    x_array = infile.x
    y_array = infile.y
    z_array = infile.z

    class02 = (cls == 2)

    df = pd.DataFrame({'X': np.fix(x_array[class02]), 'Y': np.fix(y_array[class02]), 'Z': z_array[class02]})
    grouped = df.groupby(['X', 'Y']).agg({'Z': np.median}).reset_index()

    resid = grouped.values

    xyz_txt = np.vstack((resid[:, 0], resid[:, 1], resid[:, 2]))

    np.savetxt('data.txt', resid, fmt='%s', delimiter=" ")

    class_not02 = (cls != 2)
    x_not02 = x_array[class_not02]
    y_not02 = y_array[class_not02]

    f = interpolate.SmoothBivariateSpline(resid[:, 0], resid[:, 1], resid[:, 2], s=len(x_array) * 4)

    znew = f(x_array, y_array, grid=False)

    infile.heightaboveground = z_array - znew

    return z_array - znew
