import pathmagic
from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import classify_vegetation
assert pathmagic

input_file = 'ILIJA_FlightlineTest_job081.las'
output_file = 'ILIJA_FlightlineTest_job090.las'

infile = rh_io.las_input(input_file,
                         mode='r')

outFile = rh_io.las_output(output_file, infile)

classify_vegetation(outFile)