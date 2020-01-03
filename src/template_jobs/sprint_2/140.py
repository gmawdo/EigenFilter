import pathmagic
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range
from redhawkmaster.rh_big_guns import corridor
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_file = args.input[0]
output_file = args.output[0]

infile = rh_io.las_input(input_file,
                         mode='r')

mask_corridor = corridor(infile)

point_id_not14 = las_range(dimension=infile.classification,
                           start=14, end=15,
                           reverse=True,
                           point_id_mask=mask_corridor)

outFile = rh_io.las_output(output_file, infile.point_id_not14)
