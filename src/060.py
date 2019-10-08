import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_clip

# Clip around ground points

input_file = 'ILIJA_FlightlineTest_job050.las'
output_file = 'ILIJA_FlightlineTest_job060.las'

f050 = rh_io.las_input(input_file, mode='r')
f060_000 = rh_io.las_output(output_file, f050)

point_id = np.arange(len(f050))

point_id_ground = las_range(f050.Classification,
                            start=4, end=5,
                            reverse=True,
                            point_id_mask=point_id)

rh_clip(f060_000,
        clip=100,
        cls_int=10,
        point_id_mask=point_id_ground)

point_id_noise = las_range(f060_000.Classification,
                           start=7, end=8,
                           reverse=True,
                           point_id_mask=point_id)

f060_000 = rh_io.las_output(output_file,
                            f050,
                            mask=point_id_noise)
f060_000.close()
