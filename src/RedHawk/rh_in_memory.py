import numpy as np


class RedHawkVector(np.ndarray):
    def __new__(cls, shape, **data_types):
        for key in data_types:
            if key == "points":
                raise ValueError("You cannot have a dimension called points")
        return super().__new__(shape, list(data_types.items()))

    def __init__(self, shape, **data_types):
        super().__init__(shape, list(data_types.items()))
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
        return new_vector


class RedHawkObject:
    def __init__(self, shape, **data_types):
        self.__dict__["__vector"] = RedHawkVector(shape, **data_types)
        self.__dict__["__core_data_types"] = data_types

    def __setattr__(self, key, value):
        if key in self.points.data_types:
            self.__dict__["__vector"][key] = value
        elif key == "points":
            new_points = np.empty(shape=self.points.shape, dtype=self.points.dtype)
            new_points[:] = value
            self.__dict__['__vector'] = new_points
        else:
            pass

    def __getattr__(self, key):
        if key == "points":
            return self.__dict__["__vector"]
        elif key in self.points.data_types:
            return self.points[key]
        elif key == "shape":
            return self.points.shape
        else:
            pass

    def __delattr__(self, key):
        if key in self.points.data_types and key not in self.__dict__["__core_data_types"]:
            new_points = self.points.delete_dimension(key)
            self.__dict__["__vector"] = new_points
        else:
            pass

    def __getitem__(self, subscript):
        new_vector = self.points[subscript]
        new_shape = new_vector.shape
        new_object = RedHawkVector(new_shape, **self.points.data_types)
        new_object.points = new_vector
        return new_object


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
