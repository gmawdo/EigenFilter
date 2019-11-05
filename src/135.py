from redhawkmaster import rh_io
from redhawkmaster.las_modules import virus

input_file = 'ILIJA_FlightlineTest_job081.las'
output_file = 'ILIJA_FlightlineTest_job090.las'

infile = rh_io.las_input(input_file,
                         mode='r')

outFile = rh_io.las_output(output_file, infile)

virus(outFile, clip=0.25,
      num_itter=1,
      classif=14)
