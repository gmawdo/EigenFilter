import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range
from redhawkmaster.rh_io import script_params

assert pathmagic

# This job gets one input file and outputs a file
# which has everything that is NOT classification 10
args = script_params()

job = '001'
input_file = args.input[0]
output_file = args.output[0]

f000 = rh_io.las_input(input_file, mode='r')

point_id = np.arange(len(f000))

# Select everything that is not classification 10
mask = las_range(f000.classification,
                 start=10,
                 end=11,
                 reverse=True,
                 point_id_mask=point_id)

f001 = rh_io.las_output(output_file, f000, mask)

f001.close()
