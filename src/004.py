from redhawkmaster.rh_io import merge

# This job is merging two las files

job = '004'
input_array = ['ILIJA_FlightlineTest_job001.las', 'ILIJA_FlightlineTest_job003_intensity_snapshot.las']
output_file = 'ILIJA_FlightlineTest_job004.las'

merge(array_input=input_array,
      output=output_file)
