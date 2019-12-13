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

    cluster_labels_v01_2(infile='T000_011.las',
                         outfile='T000_014.las',
                         attribute='dimension1d2d3d',
                         range_to_cluster=[[..., -1], [3]],
                         distance=0.5,
                         min_pts=1,
                         cluster_attribute='whatever',
                         minimum_length=2)

    input_file = 'T000_014.las'
    output_file = 'T000_015.las'

    ferry_v01_0(infile=input_file,
                outfile=output_file,
                attribute1='whatever',
                attribute2='intensity',
                renumber=True,
                start=0)

    input_file = 'T000_011.las'
    output_file = 'T000_016.las'

    ferry_v01_0(infile=input_file,
                outfile=output_file,
                attribute1='dimension1d2d3d',
                attribute2='intensity',
                renumber=False)


def decimation_testing():
    infile = 'T000_004.las'
    outfile = 'T001_005.las'
    decimated_outfile = 'T002_005.las'
    decimate_v01_0(infile,
                   outfile,
                   decimated_outfile,
                   voxel_size=0.01,
                   inverter_attribute="inverter")

    infile = 'T002_005.las'
    outfile = 'T002_006.las'
    dimension1d2d3d_v01_0(infile,
                          outfile)

    infile = 'T002_006.las'
    outfile = 'T002_007.las'
    eigencluster_labels_v01_2(infile,
                              outfile,
                              eigenvector_number=0,
                              attribute='dimension1d2d3d',
                              range_to_cluster=[[2]],
                              distance=0.5,
                              min_pts=1,
                              cluster_attribute='whatever',
                              minimum_length=2)

    infile = 'T002_007.las'
    outfile = 'T002_008.las'

    eigenvector_corridors(infile,
                          outfile,
                          attribute_to_corridor='whatever',
                          range_to_corridor=[[1, ...]],
                          protection_attribute='dimension1d2d3d',
                          range_to_protect=[[3]],
                          classification_of_corridor=1,
                          radius_of_cylinders=0.5,
                          length_of_cylinders=2.0)

    infile_with_inv = 'T001_005.las'
    infile_decimated = 'T002_008.las'
    outfile = "T001_015.las"
    undecimate_v01_0(infile_with_inv,
                     infile_decimated,
                     outfile,
                     inverter_attribute="inverter",
                     attributes_to_copy=["whatever", "dimension1d2d3d", "raw_classification"])

    infile = "T001_015.las"
    outfile = "T001_016.las"
    ferry_v01_0(infile=infile,
                outfile=outfile,
                attribute1='whatever',
                attribute2='intensity',
                renumber=True)

    infile = "T001_015.las"
    outfile = "T001_017.las"
    ferry_v01_0(infile=infile,
                outfile=outfile,
                attribute1='raw_classification',
                attribute2='intensity',
                renumber=False)

    infile = "T001_015.las"
    outfile = "T001_018.las"
    ferry_v01_0(infile=infile,
                outfile=outfile,
                attribute1='dimension1d2d3d',
                attribute2='intensity',
                renumber=False)

    infile = "T001_015.las"
    outfile = "T001_019.las"
    ferry_v01_1(infile=infile,
                outfile=outfile,
                attribute1='eig22',
                attribute2='intensity',
                renumber=False,
                manipulate=lambda x: 500 * (x + 1))


def build_add_classification():
    infile = 'T000_004.las'
    outfile = 'T000_005.las'
    dimension1d2d3d_v01_0(infile,
                          outfile)

    infile = 'T000_005.las'
    outfile = 'T000_006.las'
    attributeupdate_v01_0(infile,
                          outfile,
                          select_attribute='eig2',
                          select_range=[[..., 0.0]],
                          protect_attribute=None,
                          protect_range=None,
                          attack_attribute='raw_classification',
                          value=7)

    infile = 'T000_006.las'
    outfile = 'T000_007.las'
    eigencluster_labels_v01_2(infile,
                              outfile,
                              eigenvector_number=2,
                              attribute='dimension1d2d3d',
                              range_to_cluster=[[1]],
                              distance=0.5,
                              min_pts=1,
                              cluster_attribute='cluster1',
                              minimum_length=2)

    infile = 'T000_007.las'
    outfile = 'T000_008.las'
    eigenvector_corridors_v01_0(infile,
                                outfile,
                                attribute_to_corridor='cluster1',
                                range_to_corridor=[[1, ...]],
                                protection_attribute='raw_classification',
                                range_to_protect=[[7]],
                                classification_of_corridor=1,
                                radius_of_cylinders=0.5,
                                length_of_cylinders=2.0)


    infile = 'T000_008.las'
    outfile = 'T000_009.las'
    eigencluster_labels_v01_2(infile,
                              outfile,
                              eigenvector_number=0,
                              attribute='dimension1d2d3d',
                              range_to_cluster=[[2]],
                              distance=0.5,
                              min_pts=1,
                              cluster_attribute='cluster2',
                              minimum_length=2)

    infile = 'T000_009.las'
    outfile = 'T000_010.las'
    virus_v01_0(infile,
                outfile,
                distance=0.5,
                num_itter=1,
                virus_attribute='cluster2',
                virus_range=[[1, ...]],
                select_attribute=None,
                select_range=None,
                protect_attribute='raw_classification',
                protect_range=[[1], [7]],
                attack_attribute='raw_classification',
                value=2)

    infile = 'T000_010.las'
    outfile = 'T000_011.las'
    attributeupdate_v01_0(infile,
                          outfile,
                          select_attribute='dimension1d2d3d',
                          select_range=[[3]],
                          protect_attribute='raw_classification',
                          protect_range=[[1], [2], [7]],
                          attack_attribute='raw_classification',
                          value=3)

    infile = 'T000_011.las'
    outfile = 'T000_012.las'
    cluster_labels_v01_2(infile,
                         outfile,
                         attribute='raw_classification',
                         range_to_cluster=[[3]],
                         distance=0.5,
                         min_pts=1,
                         cluster_attribute='cluster3',
                         minimum_length=2.0)

    infile = 'T000_012.las'
    outfile = 'T000_013.las'
    attributeupdate_v01_0(infile,
                          outfile,
                          select_attribute='raw_classification',
                          select_range=[[3]],
                          protect_attribute='cluster3',
                          protect_range=[[1, ...]],
                          attack_attribute='raw_classification',
                          value=0)

    infile = 'T000_013.las'
    outfile = 'T000_014.las'
    virus_v01_0(infile,
                outfile,
                distance=0.5,
                num_itter=1,
                virus_attribute='raw_classification',
                virus_range=[[3]],
                select_attribute=None,
                select_range=None,
                protect_attribute='raw_classification',
                protect_range=[[1,2],[7]],
                attack_attribute='raw_classification',
                value=3)


    infile = 'T000_014.las'
    outfile = 'T000_015.las'
    virus_v01_0(infile,
                outfile,
                distance=0.5,
                num_itter=1,
                virus_attribute='raw_classification',
                virus_range=[[1, ...]],
                select_attribute='raw_classification',
                select_range=[[0]],
                protect_attribute='raw_classification',
                protect_range=[[7]],
                attack_attribute='raw_classification',
                value='auto')


# triangulation_test()
# dimension1d2d3d_clustering_testing()
# cluster_labels_testing()
# decimation_testing()
build_add_classification()
