import pathmagic

from redhawkmaster.rh_dean import delaunay_triangulation_v01_0, cluster_labels_v01_0, count_v01_0

assert pathmagic

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
                             output_ply=False)

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
