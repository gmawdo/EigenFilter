import numpy as np
from inspect import signature


class PointCloud:

    def __init__(self, values, user_info=None):
        assert values, "Provide some values."
        assert all((isinstance(v, np.ndarray) for v in values.values())), "All values must be nd arrays"
        self.user_info = user_info
        self.dimensions = set(values.keys())
        self.data_types = {key: values[key].dtype for key in self.dimensions}
        self.shapes = {key: values[key].shape for key in self.dimensions}
        for key in self.dimensions:
            setattr(self, key, values[key])

    def __len__(self):
        return {key: values[key].shape for key in self.dimensions}

    def __setattr__(self, dimension, value):
        if dimension in self.dimensions:
            super().__setattr__(dimension, value.astype(self.data_types[dimension]))
        else:
            super().__setattr__(dimension, value)

    def get_dimension(self, dimension):
        assert dimension in self.dimensions, "No attribute called " + dimension + "."
        return getattr(self, dimension)

    def set_dimension(self, dimension, value):
        assert dimension in self.dimensions, "No attribute called " + dimension + "."
        setattr(self, dimension, value.astype(data_types[dimension]))
        return None


def point_cloud_type(name, dimensions, data_types=None):
    """

    @param name:
    @param dimensions:
    @param data_types:
    @return:
    """
    if data_types:
        assert set(dimensions) == set(data_types.keys()), "Data type keys must be the attributes."
    assert dimensions, "No attributes given."

    def __init__(self, values, user_info=None):
        assert set(dimensions) == set(values.keys()), "Value keys must be the attributes."
        l = values[list(values)[0]].size
        assert all(values[key].shape == (l,) for key in values), f"All values must be 1d arrays with same length."

        self.user_info = user_info
        self.dimensions = set(dimensions)
        if data_types:
            self.data_types = {key: data_types[key] for key in self.dimensions}
        else:
            self.data_types = {key: values[key].dtype for key in self.dimensions}
        self._data_len = l
        for key in self.dimensions:
            setattr(self, key, values[key])

    def __len__(self):
        return self._data_len

    def get_dimension(self, dimension):
        assert dimension in self.dimensions, "No attribute called " + dimension + "."
        return getattr(self, dimension)

    def set_dimension(self, dimension, value):
        assert dimension in self.dimensions, "No attribute called " + dimension + "."
        setattr(self, dimension, value.astype(data_types[dimension]))
        return None

    attribute_dict = dict(__slots__=dimensions + "dimensions values user_info data_types _data_len".split(),
                          __init__=__init__,
                          __len__=__len__,
                          __setattr__=__setattr__,
                          get_dimension=get_dimension,
                          set_dimension=set_dimension)

    return type(name, (object,), attribute_dict)


RedHawkPointCloud = point_cloud_type(name="RedHawkPointCloud",
                                     dimensions="x y z classification".split(),
                                     data_types={"x": np.float64, "y": np.float64, "z": np.float64,
                                                 "classification": np.int8})


# one could choose to omit datatypes and let them be auto

def FileLaspy(filename):
    from laspy.file import File
    inFile = File(filename)
    user_info = {'inFile': inFile}
    values = {"x": inFile.x, "y": inFile.y, "z": inFile.z, "classification": inFile.classification}
    return PointCloud(values=values, user_info=user_info)


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
        i = lambda x: x
        for item in functions:
            i = item.vcomp(i)
        return i

    def run(self, infile):
        function = self.compose(infile)
