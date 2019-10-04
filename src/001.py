import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range

# This job gets one input file and outputs a file
# which has everything that is NOT classification 10

job = '001'
input_file = 'TestArea.las'
output_file = 'TestArea_job001.las'

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
