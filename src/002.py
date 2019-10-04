import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range

# This job gets one input file and outputs a file
# which has everything that IS classification 10

job = '002'
input_file = 'TestArea.las'
output_file = 'TestArea_job002.las'

f001 = rh_io.las_input(input_file, mode='r')
point_id = np.arange(len(f001))

# Select everything that IS classification 10
mask = las_range(f001.classification,
                 start=10,
                 end=11,
                 reverse=False,
                 point_id_mask=point_id)

f002 = rh_io.las_output(output_file, f001, mask)

f002.close()
