import pathmagic

from redhawkmaster.rh_dean import finish_tile
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_files = [args.input[0], args.input[1]]
output_file = args.output[0]

finish_tile(input_files, output_file)
