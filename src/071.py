import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_clip, rh_kdistance, rh_assign

# Clip around above points

input_file = 'ILIJA_FlightlineTest_job061.las'
output_file = 'ILIJA_FlightlineTest_job071.las'

f061 = rh_io.las_input(input_file, mode='r')

f061_kdistance = rh_kdistance(f061,
                              output_file_name=output_file,
                              k=1)


point_id = np.arange(len(f061_kdistance))

point_id_dropnoise = las_range(dimension=f061_kdistance.classification,
                               start=7, end=8,
                               reverse=True,
                               point_id_mask=point_id)

point_id_130_returnnum = las_range(dimension=f061_kdistance.return_num[point_id_dropnoise],
                                   start=1, end=2,
                                   reverse=False,
                                   point_id_mask=point_id_dropnoise)

point_id_130_kd_clip1 = las_range(dimension=f061_kdistance.kdistance[point_id_130_returnnum],
                                  start=0, end=1.0,
                                  reverse=True,
                                  point_id_mask=point_id_130_returnnum)

f061_kdistance.Classification = rh_assign(f061_kdistance.Classification,
                                          value=7,
                                          mask=point_id_130_kd_clip1)

f061_kdistance.kdistance = rh_assign(f061_kdistance.kdistance,
                                     value=0,
                                     mask=point_id)

f061_kdistance.close()