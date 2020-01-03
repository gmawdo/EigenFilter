import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range
from redhawkmaster.rh_io import script_params

assert pathmagic

# This job gets one input file and outputs a file
# which has everything that IS classification 10
args = script_params()

job = '002'
input_file = args.input[0]
output_file = args.output[0]

f001 = rh_io.las_input(input_file,
                       mode='r')
point_id = np.arange(len(f001))

# Select everything that IS classification 10
mask = las_range(f001.classification,
                 start=10,
                 end=11,
                 reverse=False,
                 point_id_mask=point_id)

f002 = rh_io.las_output(output_file, f001, mask)

f002.close()
