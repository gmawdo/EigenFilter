import pathmagic

from redhawkmaster.rh_big_guns import pdal_smrf
from redhawkmaster.rh_dean import bbox
from redhawkmaster.rh_io import las_input

assert pathmagic

input_file = 'T000_pid.las'
# Name of the output file
output_file = 'T000_ground.las'


bbox(input_file, output_file)
