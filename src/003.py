import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, duplicate_attr, flightline_point_counter, rh_assign, virus

# More complicated job that is classifying the flight line noise
# into classification 1. It is doing two passes of flight line counter
# and that is mild and strong. As output we have the flight line  clear.

input_file = 'ILIJA_FlightlineTest_job002.las'
output_file = 'ILIJA_FlightlineTest_job003.las'
f002 = rh_io.las_input(input_file, mode='r')

f002_000 = rh_io.las_output(output_file, f002)

f002_000_prep = duplicate_attr(infile=f002_000,
                               attribute_in='intensity',
                               attribute_out='intensity_snapshot',
                               attr_descrp='Snapshot of intensity.',
                               attr_type=5)

point_id = np.arange(len(f002))

point_id_630_730 = las_range(dimension=f002_000.intensity,
                             start=630, end=731,
                             reverse=False,
                             point_id_mask=point_id)

flightline_point_counter(f002_000_prep,
                         clip=1.0,
                         nh=40,
                         mask=point_id_630_730)

point_id_060 = las_range(f002_000_prep.intensity[point_id_630_730],
                         start=40,
                         reverse=False,
                         point_id_mask=point_id_630_730)

flightline_point_counter(f002_000_prep,
                         clip=0.25,
                         nh=80,
                         mask=point_id_060)

point_id_100 = las_range(f002_000_prep.intensity[point_id_060],
                         start=80,
                         reverse=False,
                         point_id_mask=point_id_060)

point_id_020 = las_range(f002.intensity,
                         start=630, end=730,
                         reverse=True,
                         point_id_mask=point_id)

f002_000_prep.Classification = rh_assign(f002_000_prep.Classification,
                                         value=1,
                                         mask=point_id_020)

point_id_060_noise = las_range(f002_000_prep.intensity[point_id_630_730],
                               start=40,
                               reverse=True,
                               point_id_mask=point_id_630_730)

f002_000_prep.Classification = rh_assign(f002_000_prep.Classification,
                                         value=1,
                                         mask=point_id_060_noise)


point_id_090_noise = las_range(f002_000_prep.intensity[point_id_630_730],
                               start=80,
                               reverse=True,
                               point_id_mask=point_id_630_730)

f002_000_prep.Classification = rh_assign(f002_000_prep.Classification,
                                         value=1,
                                         mask=point_id_090_noise)

f002_000_prep.intensity = f002.intensity

class_merged = virus(f002_000_prep,
                     clip=0.50,
                     num_itter=1,
                     classif=10)

f002_000_prep.classification = class_merged

f002_000_prep.close()
