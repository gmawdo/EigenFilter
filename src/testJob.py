from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_attribute_compute, rh_return_index,rh_cluster
from redhawkmaster.acqusitionQC import polygon_select, fill_shape_file

infile = rh_io.las_input('als_model.las',
                         mode='r')


outFile = rh_cluster(infile, 'cluster.las', tolerance=1.0)
print(outFile.cluster_id)
outFile.close()

# polygon_select(infile,
#                resolution=10,
#                classif=15,
#                classed='polygon')
#
# fill_shape_file(filename_poly='300_03_300ppm_OpenField_House_Class14-15_Classification15_polygon',
#                 filename_lines='SO80_ohl_11kV')
