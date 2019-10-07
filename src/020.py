from redhawkmaster.rh_big_guns import rh_extract_ground

# Extraction of ground points

input_file = 'ILIJA_FlightlineTest_job010.las'
output_file = 'ILIJA_FlightlineTest_job020.las'

rh_extract_ground(inname=input_file,
                  outname=output_file,
                  slope=0.1,
                  cut=0.0,
                  window=18,
                  cell=1.0,
                  scalar=0.5,
                  threshold=0.5)
