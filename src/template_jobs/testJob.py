import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster import rh_dean
assert pathmagic

# Input las file
infile = rh_io.las_input('T000.las',
                         mode='r')

# Output run file with
# tile_name is the name of the output file
# point_id_name is the name of the dimension
# start_step is from where to start the point id
# inc_step how much to be incremented
outfile = rh_dean.rh_add_pid(infile,
                             tile_name='T000_pid3.las',
                             point_id_name='slpid',
                             start_step=1,
                             inc_step=2)

# Test print to see the point id
# change after . if you change point_id_name with the same value
print(outfile.slpid)

# Close the file
outfile.close()
