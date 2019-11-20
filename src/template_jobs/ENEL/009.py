import pathmagic

from redhawkmaster import rh_io
from redhawkmaster.rh_dean import conductor_matters_1, veg_risk

assert pathmagic
input_file = 'T000_bbox_out_008.las'
# Name of the output file
output_file = 'T000_009_12.las'

infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

conductor_matters_1(outfile,
                    epsilon=2.5,
                    classification_in=0,
                    classification_up=1,
                    distance_ground=7,
                    length_threshold=4)

veg_risk(outfile,
         classification_in=1,
         classification_veg=3,
         classification_inter=4,
         distance_veg=3)

outfile.close()
