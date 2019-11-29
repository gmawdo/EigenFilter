import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.rh_dean import point_id, veg_risk

from redhawkmaster.acqusitionQC import polygon_select
from redhawkmaster.las_modules import las_range, rh_assign
import time
assert pathmagic
start = time.time()
input_file = '/mnt/NATIONAL_GRID/000/4ZD_Vegetation_Point_Cloud.las'
output_file = '/mnt/NATIONAL_GRID/000/T001.las'

infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

outfile = point_id(infile,
                   tile_name=output_file,
                   point_id_name="slpid",
                   start_value=0,
                   inc_step=1)

mask_class_11 = las_range(dimension=outfile.classification,
                        start=11, end=12,
                        reverse=False,
                        point_id_mask=outfile.slpid)
outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=14,
                                   mask=mask_class_11)

mask_class_12 = las_range(dimension=outfile.classification,
                        start=12, end=13,
                        reverse=False,
                        point_id_mask=outfile.slpid)
outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=14,
                                   mask=mask_class_12)

mask_class_13 = las_range(dimension=outfile.classification,
                        start=13, end=14,
                        reverse=False,
                        point_id_mask=outfile.slpid)

outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=14,
                                   mask=mask_class_13)

mask_class_31 = las_range(dimension=outfile.classification,
                        start=31, end=32,
                        reverse=False,
                        point_id_mask=outfile.slpid)

outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=4,
                                   mask=mask_class_31)

mask_class_28 = las_range(dimension=outfile.classification,
                        start=28, end=29,
                        reverse=False,
                        point_id_mask=outfile.slpid)
outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=6,
                                   mask=mask_class_28)

mask_class_10 = las_range(dimension=outfile.classification,
                        start=10, end=11,
                        reverse=False,
                        point_id_mask=outfile.slpid)

outfile.classification = rh_assign(dimension=outfile.classification,
                                   value=13,
                                   mask=mask_class_10)

veg_risk(outfile,
         classification_in=14,
         classification_veg=4,
         classification_inter=15,
         distance_veg=10)

polygon_select(outfile,
               resolution=1,
               classif=15,
               classed='polygon')

# Close the output file
outfile.close()

end = time.time()
print (end-start)
