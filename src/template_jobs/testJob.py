import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import pdal_smrf
from redhawkmaster.rh_dean import point_id
assert pathmagic

# Input las file
infile = rh_io.las_input('T000.las',
                         mode='r')

outfile = point_id(infile, tile_name="T000_pid.las",
                   point_id_name="slpid",
                   start_value=0,
                   inc_step=1)

# Run the extract ground with all parameters
outfile = pdal_smrf(outfile,
                    outname='T000_ground_withextra.las',
                    extra_dims=[('slpid', 'uint64')],
                    ground_classification=2,
                    above_ground_classification=4,
                    slope=0.1,
                    cut=0.0,
                    window=18,
                    cell=1.0,
                    scalar=0.5,
                    threshold=0.5)

# Output the extra dimensions to see if it is through
print(outfile.slpid)

# Close the file
outfile.close()
