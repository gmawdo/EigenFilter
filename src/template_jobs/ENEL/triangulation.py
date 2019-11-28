import pathmagic

from redhawkmaster.rh_dean import delaunay_triangulation

assert pathmagic

input_file = 'T000_010.las'
output_file = 'T000_011.las'

delaunay_triangulation(input_file, output_file,
                       classifications_to_triangulate=[3, 4, 5],
                       classifications_to_search=[1, 0],
                       classification_out=3,
                       min_samples=1,
                       tolerance=0.5)