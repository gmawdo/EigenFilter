import numpy as np
from inspect import signature


def nd_array_setter(key, data_type):
    def setter(self, value):
        new_value = np.zeros(self.length, dtype=data_type)
        new_value[:] = value
        setattr(self, "__" + key, new_value.astype(data_type))

    return setter


def nd_array_getter(key):
    def getter(self):
        return getattr(self, "__" + key)

    return getter


def point_cloud_type(name, datatypes):
    def __init__(self, length, user_info):
        self.user_info = user_info
        self.length = length

    def __len__(self):
        return self.length

    attribute_dict = dict(datatypes=datatypes, __init__=__init__, __len__=__len__)

    for key in datatypes:
        attribute_dict[key] = property(fset=nd_array_setter(key, datatypes[key]), fget=nd_array_getter(key))

    return type(name, (object,), attribute_dict)


RedHawkPointCloud = point_cloud_type(name="RedHawkPointCloud",
                                     datatypes={"x": np.float64, "y": np.float64, "z": np.float64,
                                                "classification": np.int8})

def FileLaspy(filename):
    from laspy.file import File
    inFile = File(filename)
    pc = RedHawkPointCloud(length=len(inFile), user_info=inFile)
    pc.x = inFile.x
    pc.y = inFile.y
    pc.z = inFile.z
    pc.classification = inFile.classification
    return pc


class RedHawkPipe:
    def __init__(self, function):
        self.function = function
        self.numargs = len(signature(function).parameters)

    def vertical_composition(self, other):
        """
        This defines vertical composition of pipes.
        """

        def composition(x):
            p = other(x)
            return self.function(*p)

        return composition

    def horizontal_composition(self, other):
        """
        This defines horizontal composition of pipes.
        """

        def composition(x):
            num_args_self = self.numargs
            num_args_other = other.numargs
            length_arg = len(x)
            assert num_args_self + num_args_other == length_arg, \
                f'Expected {num_args_self}+{num_args_other}={num_args_self + num_args_other} arguments, got {length_arg}.'
            return f(x[:num_args_self]), g(x[:-num_args_other])

        return composition


class RedHawkPipeline:
    def __init__(self, pipes):
        self.input = input
        self.pipes = pipes

    def compose(self):
        initial_pipe = self.pipes[0]
        for item in self.pipes[1:]:
            composition = item.vertical_composition(initial_pipe)
        return composition

    def run(self, infile):
        self.compose(infile)

def pipe1

pipeline = RedHawkPipeLine(
    pipes = [

    ]
)
