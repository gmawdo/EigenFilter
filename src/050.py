import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, duplicate_attr, rh_assign
from redhawkmaster.rh_big_guns import rh_hag, rh_hag_smooth

# HAG and HAG Smooth

input_file = 'ILIJA_FlightlineTest_job020.las'
output_file = 'ILIJA_FlightlineTest_job050.las'

f020 = rh_io.las_input(input_file, mode='r')
f050_000 = rh_io.las_output(output_file, f020)

point_id = np.arange(len(f020))

point_id_ground_not_above = las_range(f020.Classification,
                                      start=1, end=2,
                                      reverse=True,
                                      point_id_mask=point_id)

point_id_ground_above = las_range(f020.Classification,
                                  start=1, end=2,
                                  reverse=False,
                                  point_id_mask=point_id)

f050_000.Classification = rh_assign(f050_000.Classification,
                                    value=4,
                                    mask=point_id_ground_above)

point_id_flight_line = las_range(f050_000.user_data,
                                 start=10, end=11,
                                 reverse=False,
                                 point_id_mask=point_id)

f050_000.Classification = rh_assign(f050_000.Classification,
                                    value=10,
                                    mask=point_id_flight_line)

point_id_noise = las_range(f050_000.user_data,
                           start=7, end=8,
                           reverse=False,
                           point_id_mask=point_id)

f050_000.Classification = rh_assign(f050_000.Classification,
                                    value=7,
                                    mask=point_id_noise)

hag_filename = rh_hag(output_file, output_file)
f050_000_hag = rh_io.las_input(hag_filename, mode='r')


point_id_low_points = las_range(f050_000_hag.heightaboveground,
                                start=-0.5,
                                reverse=True,
                                point_id_mask=point_id)


f050_000_hag_high_freq = duplicate_attr(infile=f050_000_hag,
                                        attribute_in='heightaboveground',
                                        attribute_out='heightaboveground_high_frequency',
                                        attr_descrp='PDAL HAG Height Above Ground.',
                                        attr_type=10)


f050_000_hag_high_freq.Classification = rh_assign(f050_000_hag_high_freq.Classification,
                                                  value=7,
                                                  mask=point_id_low_points)

point_id_hag_noise = las_range(f050_000_hag_high_freq.Classification,
                               start=7, end=8,
                               reverse=True,
                               point_id_mask=point_id)

rh_hag_smooth(f050_000_hag_high_freq, point_id_mask=point_id_hag_noise)


f050_000 = rh_io.las_output(output_file, f050_000_hag_high_freq)

f050_000_hag_high_freq.close()
f050_000_hag.close()
f050_000.close()
