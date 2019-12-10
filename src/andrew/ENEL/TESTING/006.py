import pathmagic

from redhawkmaster.rh_dean import sd_merge

assert pathmagic

input_files = ['T000_005.las', 'T000_003_ground.las']
output_file = 'T000_006.las'

sd_merge(input_files, output_file)
