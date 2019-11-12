import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import duplicate_attr, rh_return_index, rh_cluster, rh_pdal_cluster, \
    rh_cluster_median_return, rh_assign, las_range
import os

input_file = 'ILIJA_FlightlineTest_job110_01.las'
temp = 'ILIJA_FlightlineTest_job110_02_1.las'
output_file = 'ILIJA_FlightlineTest_job110_02.las'

f110_01 = rh_io.las_input(input_file, mode='r')

f110_02 = rh_io.las_output(output_file,
                           f110_01)

point_id = np.arange(len(f110_02))

f110_02 = duplicate_attr(f110_02,
                         attribute_in='raw_classification',
                         attribute_out='user_return_index',
                         attr_type=5,
                         attr_descrp='User return index.')

rh_return_index(f110_02)

f110_02 = duplicate_attr(f110_02,
                         attribute_in='raw_classification',
                         attribute_out='cluster_id',
                         attr_type=5,
                         attr_descrp='Cluster ID.')

rh_cluster(f110_02, tolerance=2.0)

f110_02 = duplicate_attr(f110_02,
                         attribute_in='X',
                         attribute_out='cluster_id_median',
                         attr_type=9,
                         attr_descrp='Cluster Id Median')

f110_02.cluster_id_median = rh_assign(dimension=f110_02.cluster_id_median,
                                      value=0,
                                      mask=point_id)

f110_02 = duplicate_attr(f110_02,
                         attribute_in='X',
                         attribute_out='cluster_id_mean',
                         attr_type=9,
                         attr_descrp='Cluster Id Mean')

f110_02.cluster_id_mean = rh_assign(dimension=f110_02.cluster_id_mean,
                                    value=0,
                                    mask=point_id)

f110_02 = rh_cluster_median_return(f110_02, temp)

point_median_11_13 = las_range(dimension=f110_02.cluster_id_median,
                               start=11, end=13,
                               reverse=False,
                               point_id_mask=point_id)

point_user_return = las_range(dimension=f110_02.user_return_index,
                              start=11, end=12,
                              reverse=True,
                              point_id_mask=point_median_11_13)

point_user_return_1 = las_range(dimension=f110_02.user_return_index,
                                start=22, end=23,
                                reverse=True,
                                point_id_mask=point_user_return)

point_user_return_2 = las_range(dimension=f110_02.user_return_index,
                                start=33, end=34,
                                reverse=True,
                                point_id_mask=point_user_return_1)

point_user_return_3 = las_range(dimension=f110_02.user_return_index,
                                start=44, end=45,
                                reverse=True,
                                point_id_mask=point_user_return_2)

point_user_return_4 = las_range(dimension=f110_02.user_return_index,
                                start=55, end=56,
                                reverse=True,
                                point_id_mask=point_user_return_3)

f110_02 = rh_io.las_output(output_file,
                           f110_01,
                           mask=point_user_return_4)

os.system('rm ' + temp)

f110_02.close()
