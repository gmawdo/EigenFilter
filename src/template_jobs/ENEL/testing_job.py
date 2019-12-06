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

    eigencluster_labels_v01_1(infile='T000_011.las',
                              outfile='T000_014.las',
                              eigenvector_number=2,
                              attribute='dimension1d2d3d',
                              range_to_cluster=[1],
                              distance=0.5,
                              min_pts=1,
                              cluster_attribute='whatever',
                              minimum_length=2.0)

    input_file = 'T000_014.las'
    output_file = 'T000_015.las'

    ferry(input_file, output_file, 'whatever', 'intensity', True)


def cluster_labels_testing():
    input_file = 'T000_004.las'
    output_file = 'T000_011.las'

    dimension1d2d3d_v01_0(input_file,
                          output_file)

    cluster_labels_v01_1(infile='T000_011.las',
                         outfile='T000_014.las',
                         attribute='dimension1d2d3d',
                         range_to_cluster=[2, 3],
                         distance=0.5,
                         min_pts=1,
                         cluster_attribute='whatever',
                         minimum_length=10.0)

    input_file = 'T000_014.las'
    output_file = 'T000_015.las'

    ferry(input_file, output_file, 'whatever', 'intensity', False)


    input_file = 'T000_011.las'
    output_file = 'T000_016.las'

    ferry(input_file, output_file, 'dimension1d2d3d', 'intensity', False)


# triangulation_test()
#dimension1d2d3d_clustering_testing()
cluster_labels_testing()