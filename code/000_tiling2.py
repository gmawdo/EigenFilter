from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_clip, las_range
from multiprocessing import cpu_count, Pool
import numpy as np
import time


def init_equal_filesize(bigFile, ss):
    global infile, digits
    infile = bigFile
    digits = ss


def tile_equal_filesize(item):
    mask = (digits == item)

    outFile = rh_io.las_output(infile.filename.split('/')[-1].split('.las')[0]+"_Tile"+str(item).zfill(3)+".las",
                               infile, mask)
    outFile.close()


def rh_tiling_equal_filesize(filename, no_tiles=10):
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


if __name__ == '__main__':
    start_time = time.time()

    rh_tiling_equal_filesize("ILIJA_FlightlineTest.las", 10)

    end_time = time.time()
    print(end_time - start_time)

