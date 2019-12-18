import numpy as np
from inspect import signature


def point_cloud_type(name, datatypes):

    def __init__(self, length, user_info):
        self.user_info = user_info
        self.length = length
        for key in datatypes:
        	setattr(self, "__"+key, np.zeros(length, dtype = datatypes[key]))

    def __len__(self):
    	return self.length

    attribute_dict = dict(datatypes = datatypes, __init__ = __init__, __len__ = __len__)
    
    for key in datatypes:
    	def fset(self, values):
    			new = np.zeros(self.length, dtype = datatypes[key])
    			new[:] = values[:]
    			setattr(self, "__"+key, new)
    	attribute_dict[key] = property(fset = fset, fget = lambda self: getattr(self, "__"+key))

    return type(name, (object,), attribute_dict)


RedHawkPointCloud = point_cloud_type(name="RedHawkPointCloud",
                                     datatypes={"x": np.float64, "y": np.float64, "z": np.float64,
                                                 "classification": np.int8})


# one could choose to omit datatypes and let them be auto

def FileLaspy(filename):
    from laspy.file import File
    inFile = File(filename)
    pc = RedHawkPointCloud(length = len(inFile), user_info = inFile)
    pc.x = inFile.x
    pc.y = inFile.y
    pc.z = inFile.z
    pc.classification = inFile.classification
    return pc


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
