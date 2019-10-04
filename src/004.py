from redhawkmaster.rh_io import merge

# This job is merging two las files

job = '004'
input_array = ['TestArea_job001.las', 'TestArea_job003_intensity_snapshot.las']
output_file = 'TestArea_job004.las'

merge(array_input=input_array,
      output=output_file)
