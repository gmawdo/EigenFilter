import pathmagic

from redhawkmaster.las_modules import las_range
from redhawkmaster.rh_io import las_input, las_output

assert pathmagic

input_file = 'T000_002.las'
# Name of the output file
output_file1 = 'T000_003_ground.las'

output_file2 = 'T000_003_non_ground.las'

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
