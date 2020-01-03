import pathmagic
from redhawkmaster import rh_io
from redhawkmaster.las_modules import virus
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_file = args.input[0]
output_file = args.output[0]

infile = rh_io.las_input(input_file,
                         mode='r')

outFile = rh_io.las_output(output_file, infile)

virus(outFile, clip=0.25,
      num_itter=1,
      classif=14)
