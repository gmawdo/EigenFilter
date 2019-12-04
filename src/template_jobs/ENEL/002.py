import pathmagic

from redhawkmaster.rh_big_guns import pdal_smrf
from redhawkmaster.rh_dean import run_pdal_ground
from redhawkmaster.rh_io import las_input

assert pathmagic

input_file = 'T000_001.las'
# Name of the output file
output_file = 'T000_002.las'


run_pdal_ground(input_file, output_file)
