import pathmagic
from redhawkmaster.rh_dean import point_id
from redhawkmaster.rh_io import las_input
from redhawkmaster.rh_big_guns import rh_tiling_gps_equal_filesize
assert pathmagic

# Currently no output directory is defined
# Save the job with the others
# move to the directory where you want to save the files
# submit job from this directory
# python3 /home/andrew/redhawk-pure-python/src/andrew/3GRCOM/TESTING/T200_01_Tiling_Time.py
# the job contains the input location
# the directory the job is run from contains the output location
rh_tiling_gps_equal_filesize(filename="/mnt/3GRCOM/100/L002-2.las",
                             no_tiles=50)



