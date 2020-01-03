import pathmagic
from redhawkmaster.rh_io import merge, script_params

assert pathmagic

# This job is merging two las files
args = script_params()

job = '004'
input_array = [args.input[0], args.input[1]]
output_file = args.output[0]

merge(array_input=input_array,
      output=output_file)
