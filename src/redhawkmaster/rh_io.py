from laspy.file import File
import numpy as np
import os
import argparse
from redhawkmaster.rh_inmemory import RedHawkPointCloud, RedHawkPipeline, RedHawkPipe


def las_input(input_name, mode):
    """
    input laspy object and return it

    :param input_name: files name
    :type input_name: string
    :param mode: mode of input the file (it can be read, write or both)
    :type mode: string
    :return  laspy object
    """
    # Reading a las file
    # las file variable
    inFile = File(input_name, mode=mode)

    return inFile


def las_output(output_name, inFile, mask=np.array([])):
    """
    output a file based on a input file and a mask

    :param output_name: output file name
    :type output_name: string
    :param inFile: laspy object from which we get the header
    :type inFile: laspy object
    :param mask: point id mask or bool mask which will trim down the file
    :type mask: numpy array or bool array
    :return  laspy object
    """
    # Outputting a las file (do not forget to close it)
    # las file output variable
    outFile = File(output_name, mode='w', header=inFile.header)

    # If the size is not zero then we apply the mask
    # if it is zero then we take the whole file
    if mask.size != 0:
        outFile.points = inFile.points[mask]
    else:
        outFile.points = inFile.points

    return outFile


def merge(array_input, output):
    """
    Merge an array of las files using PDAL merge app.

    :param array_input: array of file names
    :type array_input: string array
    :param output: output file name
    :type output: string
    :return  writes a file to the system
    """

    # Command for the system
    command = "pdal merge "

    coms = ""

    # Input the array of file names for the input
    for inp in array_input:
        coms += inp + " "

    # Complete the command
    coms += output
    command += " " + coms

    # Run the command
    os.system(command + ' --writers.las.extra_dims="all"')


def script_params():
    """
    Input and output params
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Location of the input las file.', nargs='+', required=True)
    parser.add_argument('-o', '--output', help='Location of the output las file.', nargs='+', required=True)
    args = parser.parse_args()

    return args


class ReadIn(RedHawkPointCloud):
    def __init__(self, file_name):
        in_file = File(file_name)
        super().__init__(len(in_file))
        self.x = in_file.x
        self.y = in_file.y
        self.z = in_file.z
        self.classification = in_file.classification
        self.intensity = in_file.intensity
        self.__original_header = in_file.header
        self.__original_points = in_file.points
        self.__condition = np.ones(len(in_file), dtype=bool)

    def qc(self, new_file_name):
        out_file = File(new_file_name, mode="w", header=self.__original_header)
        condition = self.__condition
        out_file.points = self.__original_points[condition]
        out_file.x = self.x[condition]
        out_file.y = self.y[condition]
        out_file.z = self.z[condition]
        out_file.classification = self.classification[condition]
        out_file.intensity = self.intensity[condition]
        out_file.close()


class UserPipe:
    def __init__(self, tool, qc=None, **parameters):
        self.pipe = RedHawkPipe(pipe_definition=tool, **parameters)
        self.qc = qc

    def run(self, in_memory):
        self.pipe.run(in_memory)
        if self.qc is None:
            pass
        else:
            in_memory.qc(self.qc)


class UserPipeline:
    def __init__(self, *pipes):
        self.pipeline = RedHawkPipeline(*pipes)

    def run(self, in_memory):
        self.pipeline.run(in_memory)
