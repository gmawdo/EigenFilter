from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_mult_attr

input_file = 'T000_job091_01_00.las'
output_file = 'T000_job091_01_21.las'

f091_001 = rh_io.las_input(input_file, mode='r')
f091_021 = rh_io.las_output(output_file, f091_001)

rh_mult_attr(f091_021)

f091_021.intensity = f091_021.plang
