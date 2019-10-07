import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, duplicate_attr, flightline_point_counter, rh_assign, virus

# Job that is classifying additional noise in the dataset

input_file = 'ILIJA_FlightlineTest_job004.las'
output_file = 'ILIJA_FlightlineTest_job010.las'
f004 = rh_io.las_input(input_file, mode='r')

point_id = np.arange(len(f004))

f010_000_prep = rh_io.las_output(output_file,
                                 inFile=f004,
                                 mask=point_id)

f010_000_prep.user_data = f004.Classification

point_id_back_class = las_range(dimension=f010_000_prep.intensity,
                                start=0,
                                reverse=True,
                                point_id_mask=point_id)

f010_000_prep.Classification = rh_assign(f010_000_prep.Classification,
                                         value=7,
                                         mask=point_id_back_class)

f010_000_prep.user_data = rh_assign(f010_000_prep.user_data,
                                    value=7,
                                    mask=point_id_back_class)

# point_id_02of02 = las_range(dimension=f010_000_prep.intensity,
#                             start=0,
#                             reverse=False,
#                             point_id_mask=point_id_back_class)

point_id_classI = las_range(dimension=f010_000_prep.z,
                            start=0,
                            reverse=True,
                            point_id_mask=point_id)

f010_000_prep.Classification = rh_assign(f010_000_prep.Classification,
                                         value=7,
                                         mask=point_id_classI)

f010_000_prep.user_data = rh_assign(f010_000_prep.user_data,
                                    value=7,
                                    mask=point_id_classI)

point_id_returnNum = las_range(dimension=f010_000_prep.return_num,
                               start=1, end=6,
                               reverse=True,
                               point_id_mask=point_id)

f010_000_prep.Classification = rh_assign(f010_000_prep.Classification,
                                         value=7,
                                         mask=point_id_returnNum)

f010_000_prep.user_data = rh_assign(f010_000_prep.user_data,
                                    value=7,
                                    mask=point_id_returnNum)

point_id_maxNum = las_range(dimension=f010_000_prep.num_returns,
                            start=1, end=11,
                            reverse=True,
                            point_id_mask=point_id)

f010_000_prep.Classification = rh_assign(f010_000_prep.Classification,
                                         value=7,
                                         mask=point_id_maxNum)

f010_000_prep.user_data = rh_assign(f010_000_prep.user_data,
                                    value=7,
                                    mask=point_id_maxNum)

# point_id_out = las_range(dimension=f010_000_prep.Classification,
#                          start=7, end=8,
#                          reverse=True,
#                          point_id_mask=point_id)


f010_000_prep.close()
