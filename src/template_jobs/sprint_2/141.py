import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_attribute_compute, rh_mult_attr, las_range
assert pathmagic

input_file = 'ILIJA_FlightlineTest_job140.las'
output_file = 'ILIJA_FlightlineTest_job141.las'
np.seterr(divide='ignore', invalid='ignore')

f140 = rh_io.las_input(input_file, mode='r')


f141 = rh_attribute_compute(f140, output_file)

point_id = np.arange(len(f141))

rh_mult_attr(f141)


point_mask_plan_reg = las_range(dimension=f141.plan_reg,
                                end=801,
                                reverse=False,
                                point_id_mask=point_id)


point_mask_eig1 = las_range(dimension=f141.eig1,
                            start=0, end=1,
                            reverse=True,
                            point_id_mask=point_mask_plan_reg)

point_mask_rank = las_range(dimension=f141.rank,
                            start=3, end=4,
                            reverse=False,
                            point_id_mask=point_mask_eig1)

point_mask_curv = las_range(dimension=f141.curv,
                            start=0, end=201,
                            reverse=False,
                            point_id_mask=point_mask_rank)

point_mask_iso = las_range(dimension=f141.iso,
                           start=600,
                           reverse=False,
                           point_id_mask=point_mask_curv)

point_mask_plang = las_range(dimension=f141.plang,
                             start=500,
                             reverse=False,
                             point_id_mask=point_mask_iso)

point_mask_lang = las_range(dimension=f141.lang,
                            end=501,
                            reverse=False,
                            point_id_mask=point_mask_plang)

f141 = rh_io.las_output(output_file, f140, mask=point_mask_lang)