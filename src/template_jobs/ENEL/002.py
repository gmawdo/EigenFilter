import pathmagic

from redhawkmaster.rh_big_guns import pdal_smrf
from redhawkmaster.rh_dean import pdal_enel
from redhawkmaster.rh_io import las_input, script_params

assert pathmagic

args = script_params()

# Name of the input file
input_file = args.input[0]
# Name of the output file
output_file = args.output[0]


# Run the extract ground with all parameters
pdal_enel(input_file, output_file)
