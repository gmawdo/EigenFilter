import pathmagic
import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import rh_mult_attr, las_range
from redhawkmaster.rh_io import script_params

assert pathmagic

args = script_params()

input_file = args.input[0]
output_file = args.output[0]

f091_001 = rh_io.las_input(input_file, mode='r')

f091_050 = rh_io.las_output(output_file, f091_001)
point_id = np.arange(0, len(f091_050) - 1)

# 091_01_00 first !

rh_mult_attr(f091_050)

point_mask_xy_lin = las_range(dimension=f091_050.xy_lin_reg,
                              start=800,
                              reverse=False,
                              point_id_mask=point_id)

point_mask_lin_reg = las_range(dimension=f091_050.lin_reg,
                               end=1001,
                               reverse=False,
                               point_id_mask=point_mask_xy_lin)

point_mask_plan_reg = las_range(dimension=f091_050.plan_reg,
                                end=1001,
                                reverse=False,
                                point_id_mask=point_mask_lin_reg)

point_mask_eig0 = las_range(dimension=f091_050.eig0,
                            end=2,
                            reverse=False,
                            point_id_mask=point_mask_plan_reg)

point_mask_eig1 = las_range(dimension=f091_050.eig1,
                            end=10001,
                            reverse=False,
                            point_id_mask=point_mask_eig0)

point_mask_eig2 = las_range(dimension=f091_050.eig2,
                            start=3,
                            reverse=False,
                            point_id_mask=point_mask_eig1)

point_mask_rank = las_range(dimension=f091_050.rank,
                            start=0, end=2,
                            reverse=False,
                            point_id_mask=point_mask_eig2)

point_mask_curv = las_range(dimension=f091_050.curv,
                            start=0, end=3.4,
                            reverse=False,
                            point_id_mask=point_mask_rank)

point_mask_iso = las_range(dimension=f091_050.iso,
                           start=570, end=651,
                           reverse=False,
                           point_id_mask=point_mask_curv)

point_mask_ent = las_range(dimension=f091_050.ent,
                           start=0, end=501,
                           reverse=False,
                           point_id_mask=point_mask_iso)

point_mask_plang = las_range(dimension=f091_050.plang,
                             start=0, end=1001,
                             reverse=False,
                             point_id_mask=point_mask_ent)

point_mask_lang = las_range(dimension=f091_050.lang,
                            start=500, end=1001,
                            reverse=False,
                            point_id_mask=point_mask_plang)

f091_050 = rh_io.las_output(output_file, f091_001, mask=point_mask_lang)
