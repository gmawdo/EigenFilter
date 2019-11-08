import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_attribute_compute

input_file = 'T000_hag.las'
in_no_noise = 'T000_job091_01_00_nonoise.las'
output_file = 'T000_job091_01_00.las'

f090_000 = rh_io.las_input(input_file, mode='r')

point_id = np.arange(len(f090_000))

point_id_dropnoise = las_range(dimension=f090_000.classification,
                               start=7, end=8,
                               reverse=True,
                               point_id_mask=point_id)

f091_01 = rh_io.las_output(in_no_noise, f090_000, mask=point_id_dropnoise)

f091_01.z = f091_01.heightaboveground

f091_02 = rh_attribute_compute(f091_01, output_file)

f090_000.close()
f091_01.close()
f091_02.close()
