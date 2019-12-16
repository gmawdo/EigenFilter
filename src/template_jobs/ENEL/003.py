import pathmagic

from redhawkmaster.las_modules import las_range
from redhawkmaster.rh_io import las_input, las_output, script_params

assert pathmagic

args = script_params()


input_file = args.input[0]
# Name of the output file
output_file1 = args.output[0]

output_file2 = args.output[1]

# Read the input file
infile = las_input(input_file,
                   mode='r')

point_id_ground = las_range(dimension=infile.classification,
                            start=6, end=7,
                            reverse=False,
                            point_id_mask=infile.slpid)

point_id_non_ground = las_range(dimension=infile.classification,
                                start=6, end=7,
                                reverse=True,
                                point_id_mask=infile.slpid)

outfile1 = las_output(output_file1,
                      infile,
                      mask=point_id_ground)
outfile1.close()

outfile2 = las_output(output_file2,
                      infile,
                      mask=point_id_non_ground)
outfile2.close()
