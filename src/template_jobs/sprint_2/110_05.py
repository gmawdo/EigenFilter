import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_kdistance, las_range
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_file = args.input[0]
output_file = args.output[0]

f110_04 = rh_io.las_input(input_file, mode='r')

f110_05 = rh_io.las_output(output_file,
                           f110_04)

point_id = np.arange(len(f110_05))

rh_kdistance(f110_05,
             k=1,
             make_dimension=False,
             mask=point_id)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2,
                            reverse=False,
                            point_id_mask=point_id)

rh_kdistance(f110_05,
             k=2,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=3,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=2,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=3,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=1,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=3,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2.6,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=3,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2.6,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=2,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=3,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=1,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=4,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2.76,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=4,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2.76,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=3,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2.6,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=2,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=3,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=1,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=5,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=4,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=5,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=4,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=4,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2.76,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=3,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2.6,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=2,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=3,
                            reverse=False,
                            point_id_mask=point_count_max)

rh_kdistance(f110_05,
             k=1,
             make_dimension=False,
             mask=point_count_max)

point_count_max = las_range(dimension=f110_05.kdistance,
                            end=2,
                            reverse=False,
                            point_id_mask=point_count_max)

f110_05 = rh_io.las_output(output_file,
                           f110_04,
                           mask=point_count_max)
