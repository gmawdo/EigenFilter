import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import rh_extract_ground

# Extraction of ground points

input_file = 'TestArea_job010.las'
output_file = 'TestArea_job020.las'

rh_extract_ground(input_file, output_file)