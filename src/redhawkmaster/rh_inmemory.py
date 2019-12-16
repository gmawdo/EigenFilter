import numpy as np
from inspect import signature


def point_cloud_type(name, attributes, datatypes):
    assert len(attributes) >= len(datatypes), "Fewer attributes than data types."
    assert len(attributes) <= len(datatypes), "Fewer data types than attributes."

    def point_cloud_init(self, length, values, user_info=None):
        assert len(attributes) >= len(values), "Fewer attributes than values."
        assert len(attributes) <= len(values), "Fewer values than attributes."
        assert all(a.size == length for a in values), f"All values must have shape {(length,)}"
        self.user_info = user_info

    attribute_dict = {}
    attribute_dict["__init__"] = LAS_init

    return type(name, (DataFrame,), attribute_dict)


RedHawkPointCloud = point_cloud_type(name="RedHawkPointCloud",
                                     attributes="x y z classification".split(),
                                     datatypes=[np.float32, np.float32, np.float32,
                                                np.int8])  # this is a subclass of DataFrame


class FileLaspy(filename):
    inFile = File(filename)
    header = inFile.header
    x = inFile.x
    y = inFile.y
    z = inFile.z
    classification = inFile.classification
    user_info = {'header': header}
    dimensions = [spec.name for spec in inFile.point_format]
    self.redhawk = RedHawkPointCloud()


class RedHawkPipe:
    def __init__(self, function):
        self.function = function
        self.numargs = len(signature(function).parameters)

    def vcomp(self, other):
        """
        This defines vertical composition of pipes.
        """

        def vcomposition(x):
            p = other(x)
            return self.function(*p)

        return vcomposition

    def hcomp(self, other):
        """
        This defines horizontal composition of pipes.
        """

        def hcomposition(x):
            nf = f.numargs
            ng = g.numargs
            lx = len(x)
            assert nf + ng == lx, \
                f'Expected {nf}+{ng}={nf + ng} arguments, got {lx}.'
            return (f(x[:num_args(f)]), g(x[:-num_args(g)]))

        return hcomposition


class RedHawkPipeline:
    def __init__(self, functions):
        self.input = input
        self.functions = functions

    def compose(self):
        return sum(functions)

    def run(self, input):
        function = self.compose()
