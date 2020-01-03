import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, duplicate_attr, flightline_point_counter, rh_assign, virus
from redhawkmaster.rh_io import script_params

assert pathmagic

# More complicated job that is classifying the flight line noise
# into classification 1. It is doing two passes of flight line counter
# and that is mild and strong. As output we have the flight line  clear.

args = script_params()

input_file = args.input[0]
output_file = args.output[0]
f002 = rh_io.las_input(input_file, mode='r')

f002_000 = rh_io.las_output(output_file, f002)

f002_000 = duplicate_attr(infile=f002_000,
                          attribute_in='intensity',
                          attribute_out='intensity_snapshot',
                          attr_descrp='Snapshot of intensity.',
                          attr_type=5)

point_id = np.arange(len(f002))

point_id_630_730 = las_range(dimension=f002_000.intensity,
                             start=630, end=730,
                             reverse=False,
                             point_id_mask=point_id)

flightline_point_counter(f002_000,
                         clip=1.0,
                         nh=40,
                         mask=point_id_630_730)

point_id_060 = las_range(f002_000.intensity,
                         start=40,
                         reverse=False,
                         point_id_mask=point_id_630_730)

flightline_point_counter(f002_000,
                         clip=0.25,
                         nh=80,
                         mask=point_id_060)

point_id_100 = las_range(f002_000.intensity,
                         start=80,
                         reverse=False,
                         point_id_mask=point_id_060)

point_id_020 = las_range(f002.intensity,
                         start=630, end=730,
                         reverse=True,
                         point_id_mask=point_id)

f002_000.Classification = rh_assign(f002_000.Classification,
                                    value=1,
                                    mask=point_id_020)

point_id_060_noise = las_range(f002_000.intensity,
                               start=40,
                               reverse=True,
                               point_id_mask=point_id_630_730)

f002_000.Classification = rh_assign(f002_000.Classification,
                                    value=1,
                                    mask=point_id_060_noise)

point_id_090_noise = las_range(f002_000.intensity,
                               start=80,
                               reverse=True,
                               point_id_mask=point_id_630_730)

f002_000.Classification = rh_assign(f002_000.Classification,
                                    value=1,
                                    mask=point_id_090_noise)

f002_000.intensity = f002.intensity

virus(f002_000,
      clip=0.50,
      num_itter=1,
      classif=10)

f002_000.close()
