from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import hough_3d

input_file = 'ILIJA_FlightlineTest_job111.las'
output_file = 'ILIJA_FlightlineTest_job120.las'

infile = rh_io.las_input(input_file,
                         mode='r')

outFile = rh_io.las_output(output_file, infile)

hough_3d(outFile)
