import numpy as np


class RedHawkVector(np.ndarray):
    __new__ = lambda cls, shape, **data_types: np.ndarray.__new__(shape, list(data_types.items()))

    def __init__(self, shape, **data_types):
        super().__init__(shape, list(data_types.items()))
        self.__original_shape = shape
        self.__core_data_types = data_types
        self.data_types = data_types

    def add_dimension(self, key, data_type):
        new_data_types = self.data_types.copy()
        new_data_types[key] = data_type
        new_vector = RedHawkVector(self.shape, **new_data_types)
        for item in self.data_types:
            new_vector[item] = self[item]
        return new_vector

    def delete_dimension(self, key):
        new_data_types = self.data_types.copy()
        del new_data_types[key]
        new_vector = RedHawkVector(self.shape, **new_data_types)
        for item in new_data_types:
            new_vector[item] = self[item]
        return new_array_manager_type


def nd_array_setter(key, data_type):
    """
    This function is used in array_manager_type to define the property corresponding to each datatype.
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
    This function is used in array_manager_type to define the property corresponding to each datatype.
    It accesses the hidden variable so that users can evaluate the property for each dimension.
    @param key: name of dimension
    @return: the fget function for this data_type, for the property class
    """

    def getter(self):
        # simply extract the value from the hidden variable
        return getattr(self, "__" + key)

    return getter


def array_manager_type(name, data_types):
    """
    This function produces the type of an array manager with a given dictionary of data types

    @param name: Name of class. See documentation for type.
    @param data_types: Dictionary of data types
    e.g. data_types = {"x": np.float64, "y": np.float64, "z": np.float64, "intensity": np.int8}
    @return: a type whose objects are initialised shape. Note that the values for the data parameters can be set later.
    """

    def __init__(self, shape):
        self.shape = shape

    def add_dimension(self, key, data_type):
        new_data_types = self.data_types
        new_data_types[key] = data_type
        new_array_manager_type = array_manager_type(name=name,
                                                    data_types=new_data_types)
        return new_array_manager_type

    def delete_dimension(self, key):
        new_data_types = self.data_types
        del new_data_types[key]
        new_array_manager_type = array_manager_type(name=name,
                                                    data_types=new_data_types)
        return new_array_manager_type

    def __getitem__(self, subscript):
        shape = (np.empty(self.shape)[subscript]).shape
        new_array_manager = type(self)(shape)
        for key in self.data_types:
            setattr(new_array_manager, key, getattr(self, key)[subscript])
        return new_array_manager

    attribute_dict = dict(__init__=__init__,
                          __getitem__=__getitem__,
                          data_types=data_types,
                          add_dimension=add_dimension,
                          delete_dimension=delete_dimension)

    for dimension in data_types:
        attribute_dict[dimension] = property(fset=nd_array_setter(dimension, data_types[dimension]),
                                             fget=nd_array_getter(dimension))

    return type(name + str(data_types), (object,), attribute_dict)


def point_cloud_type(name, data_types):
    def __init__(self, shape):
        array_manager_type_data_types = array_manager_type("ArrayManager" + name, data_types)
        self.__array_manager = array_manager_type_data_types(shape)
        self.__dict__["__core_data_types"] = data_types

    def __setattr__(self, key, value):
        if key == "__array_manager":
            self.__dict__["__array_manager"] = value
        elif key in self.__array_manager.data_types or key:
            setattr(self.__array_manager, key, value)
        else:
            pass

    def __getattr__(self, key):
        if key == "__array_manager":
            return self.__dict__["__array_manager"]
        elif key == "__core_data_types":
            return self.__dict__["__core_data_types"]
        elif key in self.__array_manager.data_types:
            return getattr(self.__array_manager, key)
        elif key == "shape":
            return self.__array_manager.shape
        else:
            pass

    def __delattr__(self, key):
        if key in self.__array_manager.data_types and key not in self.__core_data_types:
            new_array_manager = self.__array_manager.delete_dimension(key)
            delattr(new_array_manager, key)
            self.__array_manager = new_array_manager
        else:
            pass

    def __getitem__(self, subscript):
        new_array_manager = self.__array_manager[subscript]
        new_shape = new_array_manager.shape
        new_point_cloud = type(self)(new_shape)
        new_point_cloud.__array_manager = new_array_manager
        return new_point_cloud

    attribute_dict = dict(__init__=__init__,
                          __setattr__=__setattr__,
                          __delattr__=__delattr__,
                          __getattr__=__getattr__,
                          __getitem__=__getitem__)

    return type(name, (), attribute_dict)


def tree(entry_type):
    if not isinstance(entry_type, type):
        raise ValueError(f"{entry_type.__name__} is not a type")
    if not hasattr(entry_type, "__getitem__"):
        raise ValueError(f"{entry_type.__name__} is not a subscriptable type")

    def __init__(self, **kwargs):
        dict.__init__(self, {key: entry_type(kwargs[key]) for key in kwargs})
        self.structure = {item: {item: slice(None)} for item in kwargs}
        self.parents = {item: item for item in kwargs}
        self.children = {item: None for item in kwargs}

    def split(self, key, **subscripts):
        if self.children[key]:
            raise ValueError("That key already contains split data - you can't split it again")
        intersection = {item for item in subscripts}.intersection({item for item in self})
        if intersection:
            raise ValueError(f"{intersection} already in use as sub-object(s)")
        del intersection
        self[key] = {item: self[key][subscripts[item]] for item in subscripts}
        for item in subscripts:
            self.parents[item] = key
            self.children[key] = {entry for entry in subscripts}
        self.structure[key] =

    def merge(self, key):

    return type("TreeOf" + str(entry_type.__name__),
                (dict,),
                dict(__init__=__init__,
                     split=split))


RedHawkPointCloud = point_cloud_type(name="RedHawkPointCloud",
                                     data_types={"x": np.float64, "y": np.float64, "z": np.float64,
                                                 "classification": np.uint8, "intensity": np.uint16})

RedHawkTree = tree(RedHawkPointCloud)


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
