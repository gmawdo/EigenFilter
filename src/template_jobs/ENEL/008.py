import pathmagic

from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import pdal_smrf

from redhawkmaster.rh_dean import bbox, bbox_rectangle, corridor_2d, voxel_2d, recover_un

from redhawkmaster.rh_io import las_input

assert pathmagic

input_file = 'T000_bbox_in.las'
# Name of the output file
output_file = 'T000_bbox_out_008.las'


infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

bbox_rectangle(infile,
               outfile,
               classification_in=2,
               accuracy=1000)
corridor_2d(outfile,
            distance_threshold=1,
            angle_threshold=0.2,
            classification_cond=1,
            classification_pyl=5,
            classification_up=0)

voxel_2d(outfile,
         height_threshold=5,
         classification_in=5,
         classification_up=0)

recover_un(outfile,
           accuracy=1000,
           classification_up=0,
           classification_in=5)

outfile.close()
