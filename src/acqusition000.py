import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.acqusitionQC import voxel_count, radial_count_v1, radial_count_v2, radial_count_v4
from redhawkmaster.las_modules import las_range, rh_assign

input_file = 'ILIJA_FlightlineTest_Acqusition_small.las'
output_file = 'ILIJA_FlightlineTest_Acqusition_small_1.las'


infile = rh_io.las_input(input_file, mode='r')
outfile = rh_io.las_output(output_file, infile)

point_id = np.arange(len(infile))

point_no_flight_line = las_range(dimension=outfile.Classification,
                                 start=10, end=11,
                                 reverse=True,
                                 point_id_mask=point_id)


outfile = rh_io.las_output(output_file, infile, mask=point_no_flight_line)

point_id = np.arange(len(outfile))

point_id_num_returns_1 = las_range(dimension=outfile.return_num,
                                   start=1, end=2,
                                   point_id_mask=point_id)

point_id_return_num_1 = las_range(dimension=outfile.num_returns[point_id_num_returns_1],
                                  start=1, end=2,
                                  point_id_mask=point_id_num_returns_1)

outfile.Classification = rh_assign(outfile.Classification,
                                   value=20,
                                   mask=point_id_return_num_1)

point_id_num_returns_2 = las_range(dimension=outfile.return_num,
                                   start=2, end=3,
                                   point_id_mask=point_id)

point_id_return_num_2 = las_range(dimension=outfile.num_returns[point_id_num_returns_2],
                                  start=2, end=3,
                                  point_id_mask=point_id_num_returns_2)

outfile.Classification = rh_assign(outfile.Classification,
                                   value=20,
                                   mask=point_id_return_num_2)

point_id_num_returns_3 = las_range(dimension=outfile.return_num,
                                   start=3, end=4,
                                   point_id_mask=point_id)

point_id_return_num_3 = las_range(dimension=outfile.num_returns[point_id_num_returns_3],
                                  start=3, end=4,
                                  point_id_mask=point_id_num_returns_3)

outfile.Classification = rh_assign(outfile.Classification,
                                   value=20,
                                   mask=point_id_return_num_3)

point_id_num_returns_4 = las_range(dimension=outfile.return_num,
                                   start=4, end=5,
                                   point_id_mask=point_id)

point_id_return_num_4 = las_range(dimension=outfile.num_returns[point_id_num_returns_4],
                                  start=4, end=5,
                                  point_id_mask=point_id_num_returns_4)

outfile.Classification = rh_assign(outfile.Classification,
                                   value=20,
                                   mask=point_id_return_num_4)

point_id_num_returns_5 = las_range(dimension=outfile.return_num,
                                   start=5, end=6,
                                   point_id_mask=point_id)

point_id_return_num_5 = las_range(dimension=outfile.num_returns[point_id_num_returns_5],
                                  start=5, end=6,
                                  point_id_mask=point_id_num_returns_5)

outfile.Classification = rh_assign(outfile.Classification,
                                   value=20,
                                   mask=point_id_return_num_5)

point_id_not20 = las_range(outfile.Classification,
                           start=20, end=21,
                           point_id_mask=point_id)


outfile_qc_voxel = rh_io.las_output("Ilija_QC_voxel.las", outfile, mask=point_id_not20)

outfile_qc_voxel.z = rh_assign(outfile_qc_voxel.z,
                               value=0,
                               mask=np.arange(len(outfile_qc_voxel)))

outfile_qc_voxel.intensity = voxel_count(outfile_qc_voxel,
                                         scale=1,
                                         offset=0.0)


outfile_qc_radial_v1 = rh_io.las_output("Ilija_QC_radial_v1.las", outfile, mask=point_id_not20)

outfile_qc_radial_v1.z = rh_assign(outfile_qc_radial_v1.z,
                                   value=0,
                                   mask=np.arange(len(outfile_qc_radial_v1)))

outfile_qc_radial_v1.intensity = radial_count_v1(outfile_qc_radial_v1,
                                                 radius=0.564)

outfile_qc_radial_v2 = rh_io.las_output("Ilija_QC_radial_v2.las", outfile, mask=point_id_not20)

outfile_qc_radial_v2.z = rh_assign(outfile_qc_radial_v2.z,
                                   value=0,
                                   mask=np.arange(len(outfile_qc_radial_v2)))

outfile_qc_radial_v2.intensity = radial_count_v2(outfile_qc_radial_v2,
                                                 radius=0.564)
