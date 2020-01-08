import pathmagic

# from redhawkmaster.rh_dean import *
# the above commented out to avoid conflict with rh_inmemory and acquisition_modelling
# once in-memory flow is built we should release to master - no need for version numbers anymore on functions
# note that we cannot import * locally in functions so we must do it at module lv. - hence commenting out above

from redhawkmaster.rh_inmemory import *
from redhawkmaster.rh_pipe_definitions import *
from redhawkmaster.rh_io import *

assert pathmagic



def in_memory_testing():

    UIPipeline(
        ReadIn(
            file_name="T000.las"
        ),
        ClusterLabels(
            select_attribute="intensity",
            select_range=[[0, 500], [750, 1000]],
            distance=0.5,
            min_pts=2,
            cluster_attribute="whatever",
            minimum_length=0.10
        ),
        PointId(
            point_id_name="pid",
            start_value=0,
            inc_step=1
        ),
        FerryValues(
            out_of="whatever",
            in_to="intensity"
        ),
        QC(
            file_name="QCnew.las"
        )
    )()


# triangulation_test()
# dimension1d2d3d_clustering_testing()
# cluster_labels_testing()
# decimation_testing()
# build_add_classification()
# virus_testing()
# acquisition_modeling_testing()
in_memory_testing()
