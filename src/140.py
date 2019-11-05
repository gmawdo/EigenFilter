from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range
from redhawkmaster.rh_big_guns import corridor

input_file = 'ILIJA_FlightlineTest_job081.las'
output_file = 'ILIJA_FlightlineTest_job090.las'

infile = rh_io.las_input(input_file,
                         mode='r')

mask_corridor = corridor(infile)

point_id_not14 = las_range(dimension=infile.classification,
                           start=14, end=15,
                           reverse=True,
                           point_id_mask=mask_corridor)

outFile = rh_io.las_output(output_file, infile.point_id_not14)
