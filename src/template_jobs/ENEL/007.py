import pathmagic

from redhawkmaster.rh_dean import add_hag
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_file = args.input[0]
# Name of the output file
output_file = args.output[0]

# Apply the hag
outfile = add_hag(input_file, output_file,
                  vox=1,
                  alpha=0.01)

outfile.close()
