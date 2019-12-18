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
        new_value = np.zeros(self.length, dtype=data_type)
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


def point_cloud_type(name: str, data_types: dict) -> type:
    """
    This function produces the type of a file with a given dictionary of data types

    @param name: Name of class. See documentation for type.
    @param data_types: Dictionary of data types
    e.g. data_types = {"x": np.float64, "y": np.float64, "z": np.float64, "intensity": np.int8}
    @return: a type which takes a parameters two parameters "length" (a positive integer) and "user_info" (anything)
    to initialise. Note that the values for the data parameters can be set later. Following from the above example,
    one can set inFile.x, inFile.y, inFile.z, and inFile.intensity once an object inFile is initialised.
    """

    def __init__(self, length, user_info=None):
        self.user_info = user_info
        self.length = length

    def __len__(self):
        return self.length

    attribute_dict = dict(datatypes=data_types, __init__=__init__, __len__=__len__)

    for key in data_types:
        attribute_dict[key] = property(fset=nd_array_setter(key, data_types[key]), fget=nd_array_getter(key))

    return type(name, (object,), attribute_dict)


RedHawkPointCloud = point_cloud_type(name="RedHawkPointCloud",
                                     data_types={"x": np.float64, "y": np.float64, "z": np.float64,
                                                 "classification": np.uint8, "intensity": np.uint16})


def file_laspy(filename):
    from laspy.file import File
    in_file = File(filename)
    pc = RedHawkPointCloud(length=len(in_file), user_info=in_file)
    pc.x = in_file.x
    pc.y = in_file.y
    pc.z = in_file.z
    pc.classification = in_file.classification
    pc.intensity = in_file.intensity
    return pc


class RedHawkPipeline:
    def __init__(self, pipes):
        self.pipes = pipes

    def run(self, infile):
        for item in self.pipes:
            item(infile)
