import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range

job = '002'
f001 = rh_io.las_input('ILIJA_FlightlineTest_Tile000.las', mode='r')
point_id = np.arange(len(f001))

mask = las_range(f001.classification,
                 start=10,
                 end=11,
                 reverse=False,
                 point_id_mask=point_id)

f002 = rh_io.las_output('ILIJA_FlightlineTest_Tile000_job002.las', f001, mask)

f002.close()
