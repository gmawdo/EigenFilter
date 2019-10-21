from redhawkmaster import rh_io
from redhawkmaster.acqusitionQC import polygon_select

infile = rh_io.las_input('300_03_300ppm_OpenField_House_Class14-15.las',
                         mode='r')
polygon_select(infile,
               resolution=10,
               classif=15,
               classed='vegetation')
