import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range

input_file = 'ILIJA_FlightlineTest_job090.las'
output_file = 'ILIJA_FlightlineTest_job091_01_00.las'

f090_000 = rh_io.las_input(input_file, mode='r')

point_id = np.arange(len(f090_000))

point_id_dropnoise = las_range(dimension=f090_000.classification,
                               start=7, end=8,
                               reverse=True,
                               point_id_mask=point_id)

f091_01 = rh_io.las_output(output_file, f090_000, mask=point_id_dropnoise)

f091_01.z = f091_01.heightaboveground
