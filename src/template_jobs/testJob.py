import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster import rh_dean
from redhawkmaster.rh_big_guns import extract_ground

assert pathmagic

# Input las file
infile = rh_io.las_input('T000_pid3.las',
                         mode='r')

# Run the extract ground with all parameters
outfile = extract_ground(infile,
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
print(np.sort(outfile.slpid))

# Close the file
outfile.close()
