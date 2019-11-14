import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_kdistance, rh_assign
assert pathmagic

# Clip around above points

input_file = 'ILIJA_FlightlineTest_job081.las'
output_file = 'ILIJA_FlightlineTest_job090.las'

f081 = rh_io.las_input(input_file, mode='r')

f090_000 = rh_io.las_output(output_file, f081)

point_id = np.arange(len(f090_000))

point_id_dropnoise = las_range(dimension=f090_000.classification,
                               start=7, end=8,
                               reverse=True,
                               point_id_mask=point_id)

point_id_130_returnnum = las_range(dimension=f090_000.return_num,
                                   start=1, end=2,
                                   reverse=False,
                                   point_id_mask=point_id_dropnoise)

point_id_130_num_returns = las_range(dimension=f090_000.num_returns,
                                     start=1, end=2,
                                     reverse=False,
                                     point_id_mask=point_id_130_returnnum)

point_id_130_class = las_range(dimension=f090_000.classification,
                               start=4, end=5,
                               reverse=False,
                               point_id_mask=point_id_130_num_returns)

point_id_130_intensity = las_range(dimension=f090_000.intensity,
                                   start=600,
                                   reverse=True,
                                   point_id_mask=point_id_130_class)

f090_000.Classification = rh_assign(f090_000.Classification,
                                    value=7,
                                    mask=point_id_130_intensity)

