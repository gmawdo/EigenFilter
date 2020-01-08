import pathmagic

from redhawkmaster.rh_dean import sd_merge
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_files = [args.input[0], args.input[1]]
output_file = args.output[0]


sd_merge(input_files, output_file)
