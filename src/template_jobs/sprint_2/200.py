import pathmagic
from redhawkmaster import rh_io
from redhawkmaster.rh_big_guns import extract_shape_conductors
assert pathmagic

input_file = 'ILIJA_FlightlineTest_job081.las'
output_file = 'ILIJA_FlightlineTest_job090.las'

infile = rh_io.las_input(input_file,
                         mode='r')

mask_shape = extract_shape_conductors(infile,
                                      shape_path='/home/mcus/redhawk/lidar-docker-testing/pipeline/')

outFile = rh_io.las_output(output_file, infile, mask=mask_shape)