import pathmagic

from redhawkmaster.rh_dean import add_classification, dimension1d2d3d_v01_0, attributeupdate_v01_0, \
    eigencluster_labels_v01_2, eigenvector_corridors_v01_0, virus_v01_0, cluster_labels_v01_2, create_attributes, \
    dimension1d2d3d_v01_1, attributeupdate_v01_1
from redhawkmaster.rh_io import script_params, las_input

assert pathmagic

args = script_params()

input_file = args.input[0]

# Name of the output file
output_file = args.output[0]

infile = las_input(input_file, mode='r')

outfile = create_attributes(infile,
                            output_file,
                            [("dimension1d2d3d", 6, "dimension"),
                             ("cluster1", 6, "clustering labels"),
                             ("cluster2", 6, "clustering labels"),
                             ("cluster3", 6, "clustering labels")])

dimension1d2d3d_v01_1(infile,
                      outfile)


attributeupdate_v01_1(outfile,
                      outfile,
                      select_attribute='eig2',
                      select_range=[[..., 0.0]],
                      protect_attribute=None,
                      protect_range=None,
                      attack_attribute='raw_classification',
                      value=7)

#
# eigencluster_labels_v01_2(infile,
#                           outfile,
#                           eigenvector_number=2,
#                           attribute='dimension1d2d3d',
#                           range_to_cluster=[[1]],
#                           distance=0.5,
#                           min_pts=1,
#                           cluster_attribute='cluster1',
#                           minimum_length=2)
#
# eigenvector_corridors_v01_0(infile,
#                             outfile,
#                             attribute_to_corridor='cluster1',
#                             range_to_corridor=[[1, ...]],
#                             protection_attribute='raw_classification',
#                             range_to_protect=[[7]],
#                             classification_of_corridor=1,
#                             radius_of_cylinders=0.5,
#                             length_of_cylinders=2.0)
#
# eigencluster_labels_v01_2(infile,
#                           outfile,
#                           eigenvector_number=1,  # INCORRECT NUMBER - THIS SHOULD BE A ZERO IN ANY FUTURE TEST
#                           # WHERE PARAMETER range_to_cluster==[[2]]
#                           # Eigenvector 0 is normal to the plane of best fit
#                           # Eigenvector 1 mistakenly used in ENEL trial in Oct 2019 & currently presented results 13-Dec-2019
#                           # Error approved by Andrew for Sam to ensure data quality matches client expectations
#                           attribute='dimension1d2d3d',
#                           range_to_cluster=[[2]],
#                           distance=0.5,
#                           min_pts=1,
#                           cluster_attribute='cluster2',
#                           minimum_length=2)
#
# virus_v01_0(infile,
#             outfile,
#             distance=0.5,
#             num_itter=1,
#             virus_attribute='cluster2',
#             virus_range=[[1, ...]],
#             select_attribute=None,
#             select_range=None,
#             protect_attribute='raw_classification',
#             protect_range=[[1], [7]],
#             attack_attribute='raw_classification',
#             value=2)
#
# attributeupdate_v01_0(infile,
#                       outfile,
#                       select_attribute='dimension1d2d3d',
#                       select_range=[[3]],
#                       protect_attribute='raw_classification',
#                       protect_range=[[1], [2], [7]],
#                       attack_attribute='raw_classification',
#                       value=3)
#
# cluster_labels_v01_2(infile,
#                      outfile,
#                      attribute='raw_classification',
#                      range_to_cluster=[[3]],
#                      distance=0.5,
#                      min_pts=1,
#                      cluster_attribute='cluster3',
#                      minimum_length=2.0)
#
# attributeupdate_v01_0(infile,
#                       outfile,
#                       select_attribute='raw_classification',
#                       select_range=[[3]],
#                       protect_attribute='cluster3',
#                       protect_range=[[1, ...]],
#                       attack_attribute='raw_classification',
#                       value=0)
#
# virus_v01_0(infile,
#             outfile,
#             distance=0.5,
#             num_itter=1,
#             virus_attribute='raw_classification',
#             virus_range=[[3]],
#             select_attribute=None,
#             select_range=None,
#             protect_attribute='raw_classification',
#             protect_range=[[1, 2], [7]],
#             attack_attribute='raw_classification',
#             value=3)
#
# virus_v01_0(infile,
#             outfile,
#             distance=0.5,
#             num_itter=1,
#             virus_attribute='raw_classification',
#             virus_range=[[1, ...]],
#             select_attribute='raw_classification',
#             select_range=[[0]],
#             protect_attribute='raw_classification',
#             protect_range=[[7]],
#             attack_attribute='raw_classification',
#             value='auto')
