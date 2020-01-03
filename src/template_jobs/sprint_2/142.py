import pathmagic
from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import pylon_extract
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_file = args.input[0]
output_file = args.output[0]

infile = rh_io.las_input(input_file,
                         mode='r')

outFile = rh_io.las_output(output_file, infile)

pylon_extract(outFile)
