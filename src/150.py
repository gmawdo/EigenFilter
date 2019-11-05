from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import apply_pylon

input_file = 'ILIJA_FlightlineTest_job081.las'
output_file = 'ILIJA_FlightlineTest_job090.las'

infile = rh_io.las_input(input_file,
                         mode='r')

outFile = rh_io.las_output(output_file, infile)

apply_pylon(outFile)
