import pathmagic
from redhawkmaster.rh_dean import point_id
from redhawkmaster.rh_io import las_input
from redhawkmaster.rh_big_guns import rh_tiling_gps_equal_filesize
assert pathmagic

# Name of the input file
input_file = '/mnt/3GRCOM/200/L002-2_Tile010.las'
# Name of the output file
output_file = '/mnt/3GRCOM/210/L002-2_Tile010.las'

# Read the input file
infile = las_input(input_file,
                   mode='r')

# Make output file with slpid dimension
outfile = point_id(infile,
                   tile_name=output_file,
                   point_id_name="slpid",
                   start_value=0,
                   inc_step=1)

# Select everything that is classification 10
mask_flightline = las_range(outfile.classification,
                            start=10,
                            end=11,
                            reverse=False,
                            point_id_mask=slpid)

# Close the output file
outfile.close()



