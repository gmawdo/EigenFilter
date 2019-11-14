import pathmagic
from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_mult_attr
assert pathmagic

input_file = 'T000_job091_01_00.las'
output_file = 'T000_job091_01_13.las'

f091_001 = rh_io.las_input(input_file, mode='r')
f091_013 = rh_io.las_output(output_file, f091_001)

rh_mult_attr(f091_013)

f091_013.intensity = f091_013.eig0
