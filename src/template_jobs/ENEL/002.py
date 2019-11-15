import pathmagic

from redhawkmaster.rh_big_guns import pdal_smrf
from redhawkmaster.rh_io import las_input

assert pathmagic

input_file = 'T000_pid.las'
# Name of the output file
output_file = 'T000_ground.las'

# Read the input file
infile = las_input(input_file,
                   mode='r')

# Run the extract ground with all parameters
outfile = pdal_smrf(infile,
                    outname=output_file,
                    extra_dims=[('slpid', 'uint64')],
                    ground_classification=2,
                    above_ground_classification=4,
                    slope=0.1,
                    cut=0.0,
                    window=18,
                    cell=1.0,
                    scalar=0.5,
                    threshold=0.5)

# Close the file
outfile.close()