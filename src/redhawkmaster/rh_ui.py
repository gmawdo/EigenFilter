from .rh_io import ReadIn
from .rh_inmemory import RedHawkPipeline
from .rh_pipes import *


class UIReadIn(ReadIn):
    def __init__(self, read_in: ReadIn, file_name):
        super().__init__(file_name)
        self.__stream = {}


class UIPipeline:
    def __init__(self, input_object, *pipes):
        assert isinstance(input_object,
                          ReadIn), f"The first step should read in a file, using {ReadIn.__name__}(file_name)"
        assert all(isinstance(pipe, RedHawkPipe) for pipe in
                   pipes), f"Every step after the ReadIn must be a {RedHawkPipe.__name__}."
        self.__input_object = input_object
        self.__pipeline = RedHawkPipeline(*pipes)
        self.__streams = {}

    def __call__(self):
        self.__pipeline(self.__input_object)
