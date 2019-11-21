import pathmagic

from redhawkmaster.rh_dean import add_hag

assert pathmagic
input_file = 'T000_006.las'
# Name of the output file
output_file = 'T000_007.las'

# Apply the hag
outfile = add_hag(input_file, output_file,
                  vox=1,
                  alpha=0.01)

outfile.close()
