import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.rh_dean import point_id, veg_risk

from redhawkmaster.acqusitionQC import polygon_select
from redhawkmaster.las_modules import las_range, rh_assign
import time
assert pathmagic
start = time.time()
input_file =  '/home/andrew/LIDAR/055_MEETINGS/TEMPLATES/LIVE_DEMO/TILES/RESULTS/DH5091309_000001_NoClass.las'
output_file = '/home/andrew/LIDAR/055_MEETINGS/TEMPLATES/LIVE_DEMO/TILES/RESULTS_VEG_RISK/DH5091309_000001_NoClass.las'

infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

outfile = point_id(infile,
                   tile_name=output_file,
                   point_id_name="slpid",
                   start_value=0,
                   inc_step=1)

mask_class_03 = las_range(dimension=outfile.classification,
                        start=3, end=4,
                        reverse=False,
                        point_id_mask=outfile.slpid)
outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=31,
                                   mask=mask_class_03)

mask_class_04 = las_range(dimension=outfile.classification,
                        start=4, end=5,
                        reverse=False,
                        point_id_mask=outfile.slpid)
outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=31,
                                   mask=mask_class_04)

mask_class_05 = las_range(dimension=outfile.classification,
                        start=5, end=6,
                        reverse=False,
                        point_id_mask=outfile.slpid)

outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=31,
                                   mask=mask_class_05)

veg_risk(outfile,
         classification_in=14,
         classification_veg=31,
         classification_inter=3,
         distance_veg=10)

veg_risk(outfile,
         classification_in=14,
         classification_veg=31,
         classification_inter=4,
         distance_veg=15)

veg_risk(outfile,
         classification_in=14,
         classification_veg=31,
         classification_inter=5,
         distance_veg=20)

#polygon_select(outfile,
#               resolution=1,
#               classif=15,
#               classed='polygon')

# Close the output file
outfile.close()

end = time.time()
print ('Tile01 ', end-start)
