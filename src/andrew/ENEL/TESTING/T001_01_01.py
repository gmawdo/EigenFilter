import pathmagic
from redhawkmaster.rh_dean import point_id
from redhawkmaster.rh_io import las_input, las_output
from redhawkmaster.rh_big_guns import pdal_smrf
from redhawkmaster.las_modules import las_range


import argparse
assert pathmagic

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Location of the input las file.', required=True)
parser.add_argument('-o', '--output', help='Location of the output las file.',required=True)
args = parser.parse_args()

# python3 T001_01_01.py -i /mnt/ENEL_Oct2019/000/DH5091309_000001_NoClass.las -o T000_001.las

# Name of the input file
input_file = args.input
# Name of the output file
output_file = args.output

# Read the input file
infile = las_input(input_file,
                   mode='r')

# Make output file with slpid dimension
Aaah = point_id(infile,
                   tile_name=output_file,
                   point_id_name="bob",
                   start_value=1,
                   inc_step=1)


# Run the extract ground with all parameters
outfile = pdal_smrf(Aaah,
                    outname=output_file,
                    extra_dims=[('slpid', 'uint64')],
                    ground_classification=2,
                    above_ground_classification=0,
                    slope=0.1,
                    cut=10.0,
                    window=18,
                    cell=1.0,
                    scalar=0.0,
                    threshold=0.5)


# Close the output file
outfile.close()
