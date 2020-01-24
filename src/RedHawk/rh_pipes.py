from .rh_in_memory import RedHawkPipe
from .rh_pipe_definitions import *


class Splitter(RedHawkPipe):
    def __init__(self,
                 predicates):
        super().__init__(pipe_definition=splitter,
                         predicates=predicates)


class PointId(RedHawkPipe):
    def __init__(self,
                 point_id_name,
                 start_value=0,
                 inc_step=1):
        kwargs = dict(point_id_name=point_id_name,
                      start_value=start_value,
                      inc_step=inc_step)
        super().__init__(pipe_definition=point_id,
                         **kwargs)


class ClusterLabels(RedHawkPipe):
    def __init__(self,
                 select_attribute,
                 select_range,
                 distance,
                 min_pts,
                 cluster_attribute,
                 minimum_length):
        kwargs = dict(select_attribute=select_attribute,
                      select_range=select_range,
                      distance=distance,
                      min_pts=min_pts,
                      cluster_attribute=cluster_attribute,
                      minimum_length=minimum_length)
        super().__init__(pipe_definition=cluster_labels,
                         **kwargs)


class FerryValues(RedHawkPipe):
    def __init__(self,
                 out_of,
                 in_to):
        kwargs = dict(out_of=out_of,
                      in_to=in_to)
        super().__init__(pipe_definition=ferry_values,
                         **kwargs)
