import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_attribute_compute, rh_mult_attr, las_range

input_file = 'ILIJA_FlightlineTest_job110_03.las'
output_file = 'ILIJA_FlightlineTest_job110_04.las'
np.seterr(divide='ignore', invalid='ignore')

f110_03 = rh_io.las_input(input_file, mode='r')


f110_04 = rh_attribute_compute(f110_03, output_file)

point_id = np.arange(len(f110_04))

rh_mult_attr(f110_04)

point_mask_xy_lin = las_range(dimension=f110_04.xy_lin_reg,
                              reverse=False,
                              point_id_mask=point_id)


point_mask_lin_reg = las_range(dimension=f110_04.lin_reg,
                               reverse=False,
                               point_id_mask=point_mask_xy_lin)

point_mask_plan_reg = las_range(dimension=f110_04.plan_reg,
                                reverse=False,
                                point_id_mask=point_mask_lin_reg)

point_mask_eig0 = las_range(dimension=f110_04.eig0,
                            reverse=False,
                            point_id_mask=point_mask_plan_reg)

point_mask_eig1 = las_range(dimension=f110_04.eig1,
                            reverse=False,
                            point_id_mask=point_mask_eig0)

point_mask_eig2 = las_range(dimension=f110_04.eig2,
                            start=1,
                            reverse=False,
                            point_id_mask=point_mask_eig1)

point_mask_rank = las_range(dimension=f110_04.rank,
                            start=1, end=3,
                            reverse=False,
                            point_id_mask=point_mask_eig2)

point_mask_curv = las_range(dimension=f110_04.curv,
                            reverse=False,
                            point_id_mask=point_mask_rank)

point_mask_iso = las_range(dimension=f110_04.iso,
                           start=1,
                           reverse=False,
                           point_id_mask=point_mask_curv)

point_mask_ent = las_range(dimension=f110_04.ent,
                           start=1,
                           reverse=False,
                           point_id_mask=point_mask_iso)


point_mask_plang = las_range(dimension=f110_04.plang,
                             reverse=False,
                             point_id_mask=point_mask_ent)

point_mask_lang = las_range(dimension=f110_04.lang,
                            start=700,
                            reverse=False,
                            point_id_mask=point_mask_plang)

f110_04 = rh_io.las_output(output_file, f110_03, mask=point_mask_lang)