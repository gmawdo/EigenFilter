import pathmagic
from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_mult_attr
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_file = args.input[0]
output_file = args.output[0]

f091_001 = rh_io.las_input(input_file, mode='r')
f091_010 = rh_io.las_output(output_file, f091_001)

rh_mult_attr(f091_010)

f091_010.intensity = f091_010.xy_lin_reg
