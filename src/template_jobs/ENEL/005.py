import pathmagic

from redhawkmaster.rh_dean import add_classification
from redhawkmaster.rh_io import script_params
00
assert pathmagic

args = script_params()

input_file = args.input[0]

# Name of the output file
output_file = args.output[0]

add_classification(input_file, output_file)
