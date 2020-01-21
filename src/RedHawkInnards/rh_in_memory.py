import numpy as np


class RedHawkVector(np.ndarray):

    def __new__(cls, shape, data_types):
        return super().__new__(cls, shape, list(data_types.items()))

    def add_dimension(self, key, data_type):
        dt = {key: self.dtype[key] for key in self.dtype.fields}
        new_dt = dt.copy()
        new_dt[key] = data_type
        new_vector = RedHawkVector(self.shape, new_dt)
        for item in dt:
            new_vector[item] = self[item]
        return new_vector

    def delete_dimension(self, key):
        dt = {key: self.dtype[key] for key in self.dtype.fields}
        new_dt = dt.copy()
        del new_dt[key]
        new_vector = RedHawkVector(self.shape, new_dt)
        for item in new_dt:
            new_vector[item] = self[item]
        return new_vector


class RedHawkObject:
    def __init__(self, shape, data_types):
        self.__dict__["__points"] = RedHawkVector(shape, data_types)
        self.__dict__["__core_data_types"] = data_types

    def __setattr__(self, key, value):
        if key == "points":
            new_vector = RedHawkVector(self.points.shape, self.points.dtype)
            new_vector[:] = value
            self.__dict__["__points"] = new_vector
        if key in self.points.dtype.fields:
            self.__dict__["__points"][key] = value
        else:
            pass

    def __getattr__(self, key):
        if key == "points":
            return self.__dict__["__points"]
        if key in self.points.dtype.fields:
            return self.points[key]
        else:
            pass

    def __delattr__(self, key):
        if key in self.points.dtype.fields and key not in self.__dict__["__core_data_types"]:
            self.__dict__["__points"] = self.__dict__["__points"].delete_dimension(key)
        else:
            pass

    def add_dimension(self, key, data_type):
        self.__dict__["__points"] = self.__dict__["__points"].add_dimension(key, data_type)


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


RedHawkPointCloud = lambda shape: RedHawkObject(shape,
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
