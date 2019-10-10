import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_kdistance, rh_assign

# Clip around above points

input_file = 'ILIJA_FlightlineTest_job071.las'
output_file = 'ILIJA_FlightlineTest_job081.las'

f071 = rh_io.las_input(input_file, mode='r')

point_id = np.arange(len(f071))

point_id_dropnoise = las_range(dimension=f071.classification,
                               start=7, end=8,
                               reverse=True,
                               point_id_mask=point_id)

f081_000 = rh_io.las_output(output_file,
                            f071,
                            mask=point_id_dropnoise)

rh_kdistance(f081_000,
             k=3,
             make_dimension=False)

point_id = np.arange(len(f081_000))

point_id_130_returnnum = las_range(dimension=f081_000.return_num,
                                   start=1, end=2,
                                   reverse=False,
                                   point_id_mask=point_id)

point_id_130_kd_clip1 = las_range(dimension=f081_000.kdistance[point_id_130_returnnum],
                                  start=0, end=1.3,
                                  reverse=True,
                                  point_id_mask=point_id_130_returnnum)

f081_000.Classification = rh_assign(f081_000.Classification,
                                    value=7,
                                    mask=point_id_130_kd_clip1)


f081_000.close()