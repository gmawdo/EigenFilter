from laspy.file import File
import numpy as np
import os


def las_input(input_name, mode):
    # Reading a las file
    # las file variable
    inFile = File(input_name, mode=mode)

    return inFile


def las_output(output_name, inFile, mask=np.array([])):
    # Outputting a las file (do not forget to close it)
    # las file output variable
    outFile = File(output_name, mode='w', header=inFile.header)

    if mask.size != 0:
        outFile.points = inFile.points[mask]
    else:
        outFile.points = inFile.points

    return outFile


def merge(array_input, output):

    command = "pdal merge "

    coms = ""

    for inp in array_input:
        coms += inp+" "

    coms += output
    command += " " + coms

    os.system(command+' --writers.las.extra_dims="all"')
