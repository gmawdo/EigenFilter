import numpy as np


def nd_array_setter(key, data_type):
    """
    This function is used in point_cloud_type to define the property corresponding to each datatype.
    It defines what happens when users set the property for each dimension.
    @param key: name of dimension
    @param data_type: numpy data type for that dimension
    @return: the fset function for this data_type, for the property class
    """

    def setter(self, value):
        # make a new numpy vector
        new_value = np.empty(shape=self.shape, dtype=data_type)
        # set its value
        # this syntax should tell whether we have used a value with a shape which doesn't work
        new_value[:] = value
        # set the hidden variable in such a way that we are sure the data type is preserved
        setattr(self, "__" + key, new_value.astype(data_type))

    return setter


def nd_array_getter(key):
    """
    This function is used in point_cloud_type to define the property corresponding to each datatype.
    It accesses the hidden variable so that users can evaluate the property for each dimension.
    @param key: name of dimension
    @return: the fget function for this data_type, for the property class
    """

    def getter(self):
        # simply extract the value from the hidden variable
        return getattr(self, "__" + key)

    return getter


def nd_array_deller(key):
    """
    This function is used in point_cloud_type to delete the property corresponding to each datatype.
    It accesses deletes the hidden variable so that users can evaluate the property for each dimension.
    @param key: name of dimension
    @return: the fget function for this data_type, for the property class
    """

    def deller(self):
        # simply extract the value from the hidden variable
        delattr(self, "__" + key)
        delattr(type(self), key)
        del self.data_types[key]

    return deller


def point_cloud_type(name, data_types):
    """
    This function produces the type of a file with a given dictionary of data types

    @param name: Name of class. See documentation for type.
    @param data_types: Dictionary of data types
    e.g. data_types = {"x": np.float64, "y": np.float64, "z": np.float64, "intensity": np.int8}
    @return: a type which takes a parameters two parameters "shape" (a positive integer) and "user_info" (anything)
    to initialise. Note that the values for the data parameters can be set later. Following from the above example,
    one can set inFile.x, inFile.y, inFile.z, and inFile.intensity once an object inFile is initialised.
    """

    def __init__(self, shape):
        self.shape = shape

    def add_dimension(self, key, data_type):
        new_data_types = self.data_types
        rub_out = len(str(new_data_types))
        new_data_types[key] = data_type
        new_point_cloud_type = point_cloud_type(name=self.initial_type_name[:-rub_out],
                                                data_types=new_data_types)
        self.__type__ = new_point_cloud_type
        setattr(self, key, 0)

    def __getitem__(self, indices):
        assert isinstance(indices, np.ndarray) and indices.dtype == np.bool, "Boolean indices only"
        shape = np.count_nonzero(indices)
        new_point_cloud = type(self)(shape)
        for key in self.data_types:
            setattr(new_point_cloud, key, getattr(self, key)[indices])
        return new_point_cloud

    attribute_dict = dict(__init__=__init__,
                          __getitem__=__getitem__,
                          data_types=data_types,
                          add_dimension=add_dimension)

    for dimension in data_types:
        attribute_dict[dimension] = property(fset=nd_array_setter(dimension, data_types[dimension]),
                                             fget=nd_array_getter(dimension),
                                             fdel=nd_array_deller(dimension))

    return type(name + str(data_types), (object,), attribute_dict)


RedHawkPointCloud = point_cloud_type(name="RedHawkPointCloud",
                                     data_types={"x": np.float64, "y": np.float64, "z": np.float64,
                                                 "classification": np.uint8, "intensity": np.uint16})


class RedHawkPipe:
    def __init__(self, pipe_definition, *args, **kwargs):
        self.__pipe_definition = pipe_definition
        self.__args = args
        self.__kwargs = kwargs

    def __call__(self, in_memory):
        args = self.__args
        kwargs = self.__kwargs
        return self.__pipe_definition(in_memory, *args, **kwargs)


class RedHawkPipeline:
    def __init__(self, *pipes):
        self.__pipes = pipes

    def __call__(self, *in_memory):
        for item in self.__pipes:
            item(*in_memory)
        return None


class RedHawkArrow:
    def __init__(self, source: int, target: int, arrow_definition):
        self.source = source
        self.target = target
        self.arrow_definition = arrow_definition

    def __add__(self, other):
        source = self.source + other.source
        target = self.target + other.target
        arrow_definition = lambda *x: self.arrow_definition(x[self.source]) + other.arrow_definition(x[-other.target])
        return RedHawkArrow(source, target, arrow_definition)

    def and_then(self, other):
        assert self.target == other.source, "Not composable, target and source mismatch."
        source = self.source
        target = other.target
        arrow_definition = lambda f: lambda g: lambda *x: g(f(*x))
        return RedHawkArrow(source, target, arrow_definition)
