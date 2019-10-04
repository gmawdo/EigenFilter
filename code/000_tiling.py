from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_clip, las_range
from multiprocessing import cpu_count, Pool
import numpy as np
import time


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


if __name__ == '__main__':
    start_time = time.time()

    rh_tiling_gps_equal_time("ILIJA_FlightlineTest.las", 20)

    end_time = time.time()
    print(end_time - start_time)

