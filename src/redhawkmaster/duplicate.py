import laspy

def duplicate_attr(input_name, attribute_in, attribute_out, attr_descrp, attr_type):

    inFile = laspy.file.File(input_name, mode = "r")

    outFile = laspy.file.File("T000_extradim.las", mode = "w",header = inFile.header)

    outFile.define_new_dimension(name = attribute_out ,data_type = attr_type, description = attr_descrp)

    for dimension in inFile.point_format:
        dat = inFile.reader.get_dimension(dimension.name)
        outFile.writer.set_dimension(dimension.name, dat)

    outFile.close()
    inFile = laspy.file.File("T000_extradim.las", mode = "r")
    
    outFile1 = laspy.file.File("T000_extradim1.las", mode = "w",header = inFile.header)
    


    for dimension in inFile.point_format:
        dat = inFile.reader.get_dimension(dimension.name)
        outFile1.writer.set_dimension(dimension.name, dat)
    
    in_spec = inFile.reader.get_dimension(attribute_in)
    outFile1.writer.set_dimension(attribute_out, in_spec)
    outFile1.close()

if __name__ == "__main__":
    duplicate_attr('T000.las','intensity','intensity_close','Ferry intensity',5)
