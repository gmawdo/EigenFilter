import pathmagic

from redhawkmaster.rh_dean import add_attributes
from redhawkmaster.rh_io import las_input, script_params

assert pathmagic

args = script_params()

input_file = args.input[0]

# Name of the output file
output_file = args.output[0]

add_attributes(input_file,
               output_file,
               time_intervals=10,
               k=range(4, 50),
               radius=0.5,
               virtual_speed=0,
               voxel_size=0)
