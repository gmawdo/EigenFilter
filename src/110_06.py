import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_kdistance, las_range, duplicate_attr, rh_cluster, rh_cluster_id_count_max

input_file = 'ILIJA_FlightlineTest_job110_05.las'
output_file = 'ILIJA_FlightlineTest_job110_06.las'

f110_05 = rh_io.las_input(input_file, mode='r')

f110_06 = rh_io.las_output(output_file,
                           f110_05)

point_id = np.arange(len(f110_06))

f110_06 = duplicate_attr(f110_06,
                         attribute_in='raw_classification',
                         attribute_out='cluster_id',
                         attr_type=5,
                         attr_descrp='Cluster ID.')

rh_cluster(f110_06, tolerance=2.0)

f110_06 = duplicate_attr(f110_06,
                         attribute_in='cluster_id',
                         attribute_out='cluster_id_count_max',
                         attr_type=5,
                         attr_descrp='Cluster IDCMax.')

rh_cluster_id_count_max(f110_06)

point_count_max = las_range(dimension=f110_06.cluster_id_count_max,
                            start=10,
                            reverse=False,
                            point_id_mask=point_id)

f110_06 = rh_io.las_output(output_file,
                           f110_05,
                           mask=point_count_max)

point_id = np.arange(len(f110_06))

f110_06 = duplicate_attr(f110_06,
                         attribute_in='raw_classification',
                         attribute_out='cluster_id',
                         attr_type=5,
                         attr_descrp='Cluster ID.')

rh_cluster(f110_06, tolerance=3.0)

f110_06 = duplicate_attr(f110_06,
                         attribute_in='cluster_id',
                         attribute_out='cluster_id_count_max',
                         attr_type=5,
                         attr_descrp='Cluster IDCMax.')

rh_cluster_id_count_max(f110_06)

point_count_max = las_range(dimension=f110_06.cluster_id_count_max,
                            start=20,
                            reverse=False,
                            point_id_mask=point_id)

f110_06 = rh_io.las_output(output_file,
                           f110_05,
                           mask=point_count_max)

point_id = np.arange(len(f110_06))


f110_06 = duplicate_attr(f110_06,
                         attribute_in='raw_classification',
                         attribute_out='cluster_id',
                         attr_type=5,
                         attr_descrp='Cluster ID.')

rh_cluster(f110_06, tolerance=4.0)

f110_06 = duplicate_attr(f110_06,
                         attribute_in='cluster_id',
                         attribute_out='cluster_id_count_max',
                         attr_type=5,
                         attr_descrp='Cluster IDCMax.')

rh_cluster_id_count_max(f110_06)

point_count_max = las_range(dimension=f110_06.cluster_id_count_max,
                            start=40,
                            reverse=False,
                            point_id_mask=point_id)

f110_06 = rh_io.las_output(output_file,
                           f110_05,
                           mask=point_count_max)

point_id = np.arange(len(f110_06))

f110_06 = duplicate_attr(f110_06,
                         attribute_in='raw_classification',
                         attribute_out='cluster_id',
                         attr_type=5,
                         attr_descrp='Cluster ID.')

rh_cluster(f110_06, tolerance=4.0)

f110_06 = duplicate_attr(f110_06,
                         attribute_in='cluster_id',
                         attribute_out='cluster_id_count_max',
                         attr_type=5,
                         attr_descrp='Cluster IDCMax.')

rh_cluster_id_count_max(f110_06)

point_count_max = las_range(dimension=f110_06.cluster_id_count_max,
                            start=80,
                            reverse=False,
                            point_id_mask=point_id)

f110_06 = rh_io.las_output(output_file,
                           f110_05,
                           mask=point_count_max)
