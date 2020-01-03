import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_clip
from redhawkmaster.rh_io import script_params

assert pathmagic

# Clip around above points
args = script_params()

input_file = args.input[0]
output_file = args.output[0]

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
