import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import duplicate_attr
assert pathmagic

input_file = 'ILIJA_FlightlineTest_job110_06.las'
output_file = 'ILIJA_FlightlineTest_job111.las'

f110_06 = rh_io.las_input(input_file, mode='r')

f111 = rh_io.las_output(output_file,
                           f110_06)

point_id = np.arange(len(f111))

f111 = duplicate_attr(f111,
                      attribute_in='Z',
                      attribute_out='user_z',
                      attr_type=9,
                      attr_descrp='User elevation.')

f111.z = f111.heightaboveground
