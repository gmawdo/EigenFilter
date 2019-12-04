import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.acqusitionQC import polygon_select
from redhawkmaster.las_modules import las_range, rh_assign
assert pathmagic

input_file = '<input>.las'
output_file = '<output>.las'

infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

point_id = np.arange(len(outfile))

point_id_12 = las_range(dimension=outfile.classification,
                        start=12, end=13,
                        reverse=False,
                        point_id_mask=point_id)

outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=9,
                                   mask=point_id_12)

polygon_select(infile,
               resolution=10,
               classif=15,
               classed='polygon')