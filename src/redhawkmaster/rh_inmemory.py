import numpy as np
from inspect import signature


def point_cloud_type(name, attributes, datatypes=None):
    """
    A class factory.
    @param name:
    @param attributes:
    @return:
    """
    if datatypes is not None:
        assert len(datatypes) >= len(attributes), "Fewer attributes than datatypes."
        assert len(datatypes) <= len(attributes), "Fewer datatypes than attributes."

    def __init__(self, values, user_info=None):
        L = values[0].size

        assert len(values) > 0, "No values given."
        assert len(attributes) >= len(values), "Fewer attributes than values."
        assert len(attributes) <= len(values), "Fewer values than attributes."
        assert all(a.shape == (L,) for a in values), f"All values must be 1d arrays with same length."

        self.user_info = user_info
        self.attributes = attributes
        if datatypes is None:
            self.datatypes = {item: values[index].dtype for index, item in enumerate(attributes)}
        else:
            self.datatypes = {item: datatypes[index] for index, item in enumerate(attributes)}
        self._data_len = L
        for index, item in enumerate(attributes):
            setattr(self, item, values[index].astype(self.datatypes[item]))

    def __len__(self):
        return self._data_len

    attribute_dict = dict(__slots__=attributes + "user_info attributes datatypes _data_len".split(), __init__=__init__,
                          __len__=__len__)

    return type(name, (object,), attribute_dict)


RedHawkPointCloud = point_cloud_type(name="RedHawkPointCloud",
                                     attributes="x y z classification".split())
                                     datatypes=[np.float64, np.float64, np.float64,
                                                np.int8])  # one could choose to omit datatypes and let them be auto


def FileLaspy(filename):
    from laspy.file import File
    inFile = File(filename)
    x = inFile.x
    y = inFile.y
    z = inFile.z
    classification = inFile.classification
    user_info = {'inFile': inFile}
    return RedHawkPointCloud((inFile.x, inFile.y, inFile.z, inFile.classification), user_info=user_info)


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
