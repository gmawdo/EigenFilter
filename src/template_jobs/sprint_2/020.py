import pathmagic
from redhawkmaster.rh_big_guns import rh_extract_ground
from redhawkmaster.rh_io import script_params

assert pathmagic

# Extraction of ground points
args = script_params()

input_file = args.input[0]
output_file = args.output[0]

rh_extract_ground(inname=input_file,
                  outname=output_file,
                  slope=0.1,
                  cut=0.0,
                  window=18,
                  cell=1.0,
                  scalar=0.5,
                  threshold=0.5)
