import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_clip
from redhawkmaster.rh_io import script_params

assert pathmagic

# Clip around ground points
args = script_params()

input_file = args.input[0]
output_file = args.output[0]

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
