from pandas import DataFrame
from inspect import signature


def point_cloud_type(name, attributes):
    def LAS_init(self, values, user_info=None):
        DataFrame.__init__(self, data=values, columns=attributes)
        self.user_info = user_info

    return type(name, (DataFrame,), {"__init__": LAS_init})


RedHawkPointCloud = point_cloud_type("RedHawkPointCloud",
                                     "x y z classification".split())  # this is a subclass of DataFrame


def File_laspy:
    inFile = File(filename)
    header = inFile.header
    points = inFile.points
    x = inFile.x
    y = inFile.y
    z = inFile.z
    classification = inFile.classification
    user_info = {'header': header, 'points': points}
    self.redhawk = RedHawkPointCloud()


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
