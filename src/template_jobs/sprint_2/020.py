import pathmagic
from redhawkmaster.rh_big_guns import pdal_smrf
from redhawkmaster.rh_io import script_params

assert pathmagic

# Extraction of ground points
args = script_params()

input_file = args.input[0]
output_file = args.output[0]

pdal_smrf(inname=input_file,
          outname=output_file,
          extra_dims=None,
          slope=0.1,
          cut=0.0,
          window=18,
          cell=1.0,
          scalar=0.5,
          threshold=0.5)
