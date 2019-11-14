import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, rh_kdistance
assert pathmagic

input_file = 'ILIJA_FlightlineTest_job100.las'
output_file = 'ILIJA_FlightlineTest_job110_01.las'

f100 = rh_io.las_input(input_file, mode='r')

f110_01 = rh_io.las_output(output_file, f100)

point_id = np.arange(len(f110_01))

point_highfreq = las_range(dimension=f110_01.heightaboveground_high_frequency,
                           start=3,
                           reverse=False,
                           point_id_mask=point_id)

rh_kdistance(f110_01,
             k=1,
             make_dimension=False,
             mask=point_highfreq)


point_kdist_1 = las_range(dimension=f110_01.kdistance,
                          end=1,
                          reverse=False,
                          point_id_mask=point_highfreq)

rh_kdistance(f110_01,
             k=2,
             make_dimension=False,
             mask=point_kdist_1)


point_kdist_2 = las_range(dimension=f110_01.kdistance,
                          end=2,
                          reverse=False,
                          point_id_mask=point_kdist_1)


rh_kdistance(f110_01,
             k=1,
             make_dimension=False,
             mask=point_kdist_2)


point_kdist_11 = las_range(dimension=f110_01.kdistance,
                           end=1,
                           reverse=False,
                           point_id_mask=point_kdist_2)


rh_kdistance(f110_01,
             k=3,
             make_dimension=False,
             mask=point_kdist_11)


point_kdist_25 = las_range(dimension=f110_01.kdistance,
                           end=2.5,
                           reverse=False,
                           point_id_mask=point_kdist_11)


rh_kdistance(f110_01,
             k=3,
             make_dimension=False,
             mask=point_kdist_25)


point_kdist2_25 = las_range(dimension=f110_01.kdistance,
                            end=2.5,
                            reverse=False,
                            point_id_mask=point_kdist_25)

rh_kdistance(f110_01,
             k=2,
             make_dimension=False,
             mask=point_kdist2_25)


point_kdist2_2 = las_range(dimension=f110_01.kdistance,
                           end=2,
                           reverse=False,
                           point_id_mask=point_kdist2_25)


rh_kdistance(f110_01,
             k=1,
             make_dimension=False,
             mask=point_kdist2_2)


point_kdist1_1 = las_range(dimension=f110_01.kdistance,
                           end=1,
                           reverse=False,
                           point_id_mask=point_kdist2_2)

rh_kdistance(f110_01,
             k=4,
             make_dimension=False,
             mask=point_kdist1_1)


point_kdist4 = las_range(dimension=f110_01.kdistance,
                         end=2.75,
                         reverse=False,
                         point_id_mask=point_kdist1_1)

rh_kdistance(f110_01,
             k=4,
             make_dimension=False,
             mask=point_kdist4)


point_kdist4_1 = las_range(dimension=f110_01.kdistance,
                           end=2.75,
                           reverse=False,
                           point_id_mask=point_kdist4)

rh_kdistance(f110_01,
             k=3,
             make_dimension=False,
             mask=point_kdist4_1)


point_kdist3_25 = las_range(dimension=f110_01.kdistance,
                            end=2.5,
                            reverse=False,
                            point_id_mask=point_kdist4_1)

rh_kdistance(f110_01,
             k=2,
             make_dimension=False,
             mask=point_kdist3_25)


point_kdist2_21 = las_range(dimension=f110_01.kdistance,
                            end=2,
                            reverse=False,
                            point_id_mask=point_kdist3_25)

rh_kdistance(f110_01,
             k=1,
             make_dimension=False,
             mask=point_kdist2_21)


point_kdist1_21 = las_range(dimension=f110_01.kdistance,
                            end=1,
                            reverse=False,
                            point_id_mask=point_kdist2_21)

rh_kdistance(f110_01,
             k=5,
             make_dimension=False,
             mask=point_kdist1_21)


point_kdist5_3 = las_range(dimension=f110_01.kdistance,
                           end=3,
                           reverse=False,
                           point_id_mask=point_kdist1_21)

rh_kdistance(f110_01,
             k=5,
             make_dimension=False,
             mask=point_kdist5_3)


point_kdist5_32 = las_range(dimension=f110_01.kdistance,
                            end=3,
                            reverse=False,
                            point_id_mask=point_kdist5_3)

rh_kdistance(f110_01,
             k=4,
             make_dimension=False,
             mask=point_kdist5_32)


point_kdist4_2 = las_range(dimension=f110_01.kdistance,
                           end=2.75,
                           reverse=False,
                           point_id_mask=point_kdist5_32)

rh_kdistance(f110_01,
             k=3,
             make_dimension=False,
             mask=point_kdist4_2)


point_kdist3_252 = las_range(dimension=f110_01.kdistance,
                             end=2.5,
                             reverse=False,
                             point_id_mask=point_kdist4_2)

rh_kdistance(f110_01,
             k=2,
             make_dimension=False,
             mask=point_kdist3_252)


point_kdist2_22 = las_range(dimension=f110_01.kdistance,
                            end=2,
                            reverse=False,
                            point_id_mask=point_kdist3_252)

rh_kdistance(f110_01,
             k=1,
             make_dimension=False,
             mask=point_kdist2_22)


point_kdist1_2 = las_range(dimension=f110_01.kdistance,
                           end=1,
                           reverse=False,
                           point_id_mask=point_kdist2_22)

f110_01.close()

f110_01 = rh_io.las_output(output_file, f100, point_kdist1_2)
f110_01.close()