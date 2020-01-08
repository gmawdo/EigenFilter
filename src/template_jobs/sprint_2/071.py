import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_clip, rh_kdistance, rh_assign
from redhawkmaster.rh_io import script_params

assert pathmagic

# Clip around above points
args = script_params()

input_file = args.input[0]
output_file = args.output[0]

f061 = rh_io.las_input(input_file, mode='r')

point_id = np.arange(len(f061))

point_id_dropnoise = las_range(dimension=f061.classification,
                               start=7, end=8,
                               reverse=True,
                               point_id_mask=point_id)

f061_kdistance = rh_kdistance(f061,
                              output_file_name=output_file,
                              k=1,
                              mask=point_id_dropnoise)

point_id = np.arange(len(f061_kdistance))

point_id_130_returnnum = las_range(dimension=f061_kdistance.return_num,
                                   start=1, end=2,
                                   reverse=False,
                                   point_id_mask=point_id)

point_id_130_kd_clip1 = las_range(dimension=f061_kdistance.kdistance,
                                  start=0, end=0.8,
                                  reverse=True,
                                  point_id_mask=point_id_130_returnnum)

f061_kdistance.Classification = rh_assign(f061_kdistance.Classification,
                                          value=7,
                                          mask=point_id_130_kd_clip1)

f061_kdistance.kdistance = rh_assign(f061_kdistance.kdistance,
                                     value=0,
                                     mask=point_id)

f061_kdistance.close()
