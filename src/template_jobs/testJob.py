import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.acqusitionQC import polygon_select
from redhawkmaster.las_modules import las_range, rh_assign, las_range_v01

assert pathmagic

input_file = 'ILIJA_FlightlineTest.las'
output_file = 'ILIJA_FlightlineTest_02.las'

infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

point_id = np.arange(len(outfile))

print(las_range_v01(dimension=point_id,
                    range=[(1, 20), [30, 50]],
                    inverse=False,
                    point_id_mask=point_id))
