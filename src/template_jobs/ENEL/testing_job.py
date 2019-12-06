import pathmagic

from redhawkmaster.rh_dean import *

assert pathmagic


def triangulation_test():
    input_file = 'T000_009.las'
    output_file = 'T000_011.las'

    cluster_labels_v01_0(input_file,
                         output_file,
                         classification_to_cluster=[3],
                         tolerance=0.5,
                         min_pts=1,
                         cluster_attribute="userdefinedname")

    input_file = 'T000_011.las'
    output_file = 'T000_012.las'

    delaunay_triangulation_v01_0(input_file,
                                 output_file,
                                 classifications_to_search=[1, 0],
                                 classification_out=3,
                                 cluster_attribute="userdefinedname",
                                 output_ply=True)

    input_file = 'T000_012.las'
    output_file = 'T000_013.las'

    cluster_labels_v01_0(input_file,
                         output_file,
                         classification_to_cluster=[3],
                         tolerance=0.5,
                         min_pts=1,
                         cluster_attribute="userdefinedname")

    input_file = 'T000_013.las'
    output_file = 'T000_014.las'

    count_v01_0(input_file,
                output_file,
                attribute="userdefinedname")


def dimension1d2d3d_clustering_testing():
    input_file = 'T000_004.las'
    output_file = 'T000_011.las'

    dimension1d2d3d_v01_0(input_file,
                          output_file)

    input_file = 'T000_011.las'
    output_file = 'T000_013.las'

    ferry(input_file, output_file, 'dimension1d2d3d', 'raw_classification', False)

    input_file = 'T000_013.las'
    output_file = 'T000_014.las'

    cluster_labels_v01_1(input_file,
                         output_file,
                         'intensity',
                         range(1000),
                         0.5,
                         1,
                         "whatever",
                         0.5)

    input_file = 'T000_014.las'
    output_file = 'T000_015.las'

    ferry(input_file, output_file, 'whatever', 'intensity', True)


# triangulation_test()
dimension1d2d3d_clustering_testing()
