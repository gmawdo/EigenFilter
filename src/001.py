import numpy as np
from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range

job = '001'
f000 = rh_io.las_input('TestArea.las', mode='r')

point_id = np.arange(len(f000))

mask = las_range(f000.classification,
                 start=10,
                 end=11,
                 reverse=True,
                 point_id_mask=point_id)

f001 = rh_io.las_output('TestArea_job001.las', f000, mask)

f001.close()
