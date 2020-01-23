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

    def __getitem__(self, item):
        vector = self.points[item]
        new_object = RedHawkObject(vector.shape, {key: vector.dtype[key] for key in vector.dtype.fields})
        new_object.points = vector
        return new_object

    def add_dimension(self, key, data_type):
        self.__dict__["__points"] = self.__dict__["__points"].add_dimension(key, data_type)


class RedHawkOperad:
    def __init__(self, object):
        self.arity = 1
        self.parents = {'':''}
        self.slices = {'': slice(None)}
        self.object = object

    def split(self, key, *names, **slices):
        for item in names:
            if item in self.parents:
                raise ValueError("{} already used. ('' is the root object.)".format(item))
            self.parents[item]=key

    def __getitem__(self, key):
        working_key = key
        slices = []
        while working_key != '':
            slices.append(self.slices[working_key])
            working_key = self.parents[working_key]


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
