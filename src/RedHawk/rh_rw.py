from laspy.file import File
import numpy as np
import os
import argparse
from .rh_in_memory import RedHawkPointCloud, RedHawkPipe, RedHawkPipeline


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
    in_file = File(input_name, mode=mode)

    return in_file


def las_output(output_name, in_file, mask=np.array([])):
    """
    output a file based on a input file and a mask

    :param output_name: output file name
    :type output_name: string
    :param in_file: laspy object from which we get the header
    :type in_file: laspy object
    :param mask: point id mask or bool mask which will trim down the file
    :type mask: numpy array or bool array
    :return  laspy object
    """
    # Outputting a las file (do not forget to close it)
    # las file output variable
    out_file = File(output_name, mode='w', header=in_file.header)

    # If the size is not zero then we apply the mask
    # if it is zero then we take the whole file
    if mask.size != 0:
        out_file.points = in_file.points[mask]
    else:
        out_file.points = in_file.points

    return out_file


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


# === STUFF FOR USER PIPELINE ===

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
        self.__streams = {}

    def qc(self, new_file_name, index=...):
        out_file = File(new_file_name, mode="w", header=self.__original_header)
        out_file.points = self.__original_points[index]
        out_file.classification = self.classification[index]
        out_file.intensity = self.intensity[index]
        out_file.close()


class QC(RedHawkPipe):
    def __init__(self,
                 file_name):
        super().__init__(pipe_definition=ReadIn.qc, new_file_name=file_name)
