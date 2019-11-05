from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_cluster, rh_cluster_id_count_max, virus
from redhawkmaster.acqusitionQC import polygon_select, fill_shape_file
from redhawkmaster.rh_big_guns import hough_3d, apply_hough, corridor, pylon_extract, apply_pylon, \
    extract_shape_conductors

infile = rh_io.las_input('applied_150.las',
                         mode='r')

mask_shape = extract_shape_conductors(infile,
                                      shape_path='/home/mcus/redhawk/lidar-docker-testing/pipeline/')

outFile = rh_io.las_output('shaped_200.las', infile, mask=mask_shape)

# 150 works !!!
# apply_pylon(outFile)

# 142 works!
# pylon_extract(outFile, classification=15)

# 140 works !!!
# mask = corridor(infile)
# print(mask)
# outFile = rh_io.las_output('corridor_140.las', infile,mask=mask)


#  130 Works !!!
# apply_hough(outFile)

# virus(outFile,
#       clip=0.25,
#       num_itter=1,
#       classif=14)


# outFile = rh_cluster_id_count_max(infile, 'cluster_idmax_2.las')
# print(outFile.cluster_id_count_max)
# outFile.close()

# polygon_select(infile,
#                resolution=10,
#                classif=15,
#                classed='polygon')
#
# fill_shape_file(filename_poly='300_03_300ppm_OpenField_House_Class14-15_Classification15_polygon',
#                 filename_lines='SO80_ohl_11kV')
