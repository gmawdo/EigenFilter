import pathmagic

from redhawkmaster.rh_dean import add_attributes
from redhawkmaster.rh_io import las_input

assert pathmagic

input_file = 'T000_003_non_ground.las'

# Name of the output file
output_file = 'T000_004.las'

add_attributes(input_file,
               output_file,
               time_intervals=10,
               k=range(4, 50),
               radius=0.5,
               virtual_speed=0,
               voxel_size=0)
