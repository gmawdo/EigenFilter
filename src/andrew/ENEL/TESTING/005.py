import pathmagic

from redhawkmaster.rh_dean import add_classification

assert pathmagic
input_file = 'T000_004.las'

# Name of the output file
output_file = 'T000_005.las'

add_classification(input_file, output_file)