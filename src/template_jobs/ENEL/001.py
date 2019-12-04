import pathmagic
from redhawkmaster.rh_dean import point_id
from redhawkmaster.rh_io import las_input, script_params
assert pathmagic

args = script_params()

# Name of the input file
input_file = args.input[0]
# Name of the output file
output_file = args.output[0]

# Read the input file
infile = las_input(input_file,
                   mode='r')

# Make output file with slpid dimension
outfile = point_id(infile,
                   tile_name=output_file,
                   point_id_name="slpid",
                   start_value=0,
                   inc_step=1)

# Close the output file
outfile.close()
