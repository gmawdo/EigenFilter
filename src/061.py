import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_clip

# Clip around above points

input_file = 'ILIJA_FlightlineTest_job060.las'
output_file = 'ILIJA_FlightlineTest_job061.las'

f060 = rh_io.las_input(input_file, mode='r')
f061_000 = rh_io.las_output(output_file, f060)

point_id = np.arange(len(f060))

point_id_ground = las_range(f060.Classification,
                            start=2, end=3,
                            reverse=True,
                            point_id_mask=point_id)

rh_clip(f061_000,
        clip=100,
        cls_int=10,
        point_id_mask=point_id_ground)

point_id_above = las_range(f061_000.Classification,
                           start=4, end=5,
                           reverse=False,
                           point_id_mask=point_id)

f061_000 = rh_io.las_output(output_file,
                            f060,
                            mask=point_id_above)
f061_000.close()
