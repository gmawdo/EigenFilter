import pathmagic

from redhawkmaster.rh_dean import add_hag

assert pathmagic
input_file = 'T000_ground.las'
# Name of the output file
output_file = 'T000_hag.las'

# Apply the hag
outfile = add_hag(input_file, output_file,
                  vox=1,
                  alpha=0.01)

# Redatum the data. Maybe we don't need this
outfile.z = outfile.hag

outfile.close()
