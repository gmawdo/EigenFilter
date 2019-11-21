import pathmagic

from redhawkmaster.rh_dean import sd_merge

assert pathmagic

input_files = ['T000_non_ground_points.las', 'T000_ground_points.las']
output_file = 'T000_006.las'

sd_merge(input_files, output_file)
