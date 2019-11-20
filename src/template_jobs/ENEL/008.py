import pathmagic

from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import pdal_smrf

from redhawkmaster.rh_dean import bbox, bbox_rectangle, corridor_2d, voxel_2d, recover_un

from redhawkmaster.rh_io import las_input

assert pathmagic

input_file = 'T000_bbox_out_tool3.las'
# Name of the output file
output_file = 'T000_bbox_out_tool4.las'


infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

# classn = bbox_rectangle(infile,
#                         outfile,
#                         classification_in=2,
#                         accuracy=1000)

# voxel_2d(outfile,
#          height_threshold=5,
#          classification_in=5,
#          classification_un=0)

recover_un(outfile,
           classification_un=0,
           classification_pyl=5)





outfile.close()

# bbox(input_file, output_file)
