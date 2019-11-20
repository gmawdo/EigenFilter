import pathmagic

from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import pdal_smrf
from redhawkmaster.rh_dean import bbox, bbox_rectangle, corridor_2d
from redhawkmaster.rh_io import las_input

assert pathmagic

input_file = 'T000_bbox_in.las'
# Name of the output file
output_file = 'T000_bbox_out_tool2.las'

infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

classn = bbox_rectangle(infile,
                        outfile,
                        classification_in=2,
                        accuracy=1000)

corridor_2d(outfile,
            distance=1,
            angle_th=0.2)

outfile.close()

# bbox(input_file, output_file)
