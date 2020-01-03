import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import duplicate_attr, rh_cluster, las_range, rh_cluster_id_count_max
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_file = args.input[0]
output_file = args.output[0]

f110_02 = rh_io.las_input(input_file, mode='r')

f110_03 = rh_io.las_output(output_file,
                           f110_02)

point_id = np.arange(len(f110_03))

f110_03 = duplicate_attr(f110_03,
                         attribute_in='raw_classification',
                         attribute_out='cluster_id',
                         attr_type=5,
                         attr_descrp='Cluster ID.')

rh_cluster(f110_03, tolerance=1.0)

f110_03 = duplicate_attr(f110_03,
                         attribute_in='cluster_id',
                         attribute_out='cluster_id_count_max',
                         attr_type=5,
                         attr_descrp='Cluster IDCMax.')

rh_cluster_id_count_max(f110_03)

point_count_max = las_range(dimension=f110_03.cluster_id_count_max,
                            start=3,
                            reverse=False,
                            point_id_mask=point_id)

f110_03 = rh_io.las_output(output_file,
                           f110_02,
                           mask=point_count_max)

point_id = np.arange(len(f110_03))

f110_03 = duplicate_attr(f110_03,
                         attribute_in='raw_classification',
                         attribute_out='cluster_id',
                         attr_type=5,
                         attr_descrp='Cluster ID.')

rh_cluster(f110_03, tolerance=2.0)

f110_03 = duplicate_attr(f110_03,
                         attribute_in='cluster_id',
                         attribute_out='cluster_id_count_max',
                         attr_type=5,
                         attr_descrp='Cluster IDCMax.')

rh_cluster_id_count_max(f110_03)

point_count_max = las_range(dimension=f110_03.cluster_id_count_max,
                            start=6,
                            reverse=False,
                            point_id_mask=point_id)

f110_03 = rh_io.las_output(output_file,
                           f110_02,
                           mask=point_count_max)

point_id = np.arange(len(f110_03))

f110_03 = duplicate_attr(f110_03,
                         attribute_in='raw_classification',
                         attribute_out='cluster_id',
                         attr_type=5,
                         attr_descrp='Cluster ID.')

rh_cluster(f110_03, tolerance=4.0)

f110_03 = duplicate_attr(f110_03,
                         attribute_in='cluster_id',
                         attribute_out='cluster_id_count_max',
                         attr_type=5,
                         attr_descrp='Cluster IDCMax.')

rh_cluster_id_count_max(f110_03)

point_count_max = las_range(dimension=f110_03.cluster_id_count_max,
                            start=10,
                            reverse=False,
                            point_id_mask=point_id)

f110_03 = rh_io.las_output(output_file,
                           f110_02,
                           mask=point_count_max)
