from laspy.file import File

# Function which will range the specific tile
# from the values start and end on a specific dimension
def range(inFile,dimension,start,end):
    # Reading the file
    # Please note the location of the file
    #inFile = File(tile_name, mode = 'r')

    # For each dimension we make a mask from start until end
    # If a dimension is in lowercase letters is scaled one
    if dimension == 'X':
        mask = (inFile.X >= start) & (inFile.X <= end)
    if dimension == 'Y':
        mask = (inFile.Y >= start) & (inFile.Y <= end)
    if dimension == 'Z':
        mask = (inFile.Z >= start) & (inFile.Z <= end)
    if dimension == 'x':
        mask = (inFile.x >= start) & (inFile.x <= end)
    if dimension == 'y':
        mask = (inFile.y >= start) & (inFile.y <= end)
    if dimension == 'z':
        mask = (inFile.z >= start) & (inFile.z <= end)
    if dimension == 'Classification':
        mask = (inFile.Classification >= start) & (inFile.Classification <= end)
    if dimension == 'heightaboveground':
        mask = (inFile.heightaboveground >= start) & (inFile.heightaboveground <= end)
    if dimension == 'intensity':
        mask = (inFile.intensity >= start) & (inFile.intensity <= end)
    if dimension == 'flag_byte':
        mask = (inFile.flag_byte >= start) & (inFile.flag_byte <= end)
    if dimension == 'raw_classification':
        mask = (inFile.raw_classification >= start) & (inFile.raw_classification <= end)
    if dimension == 'scan_angle_rank':
        mask = (inFile.scan_angle_rank >= start) & (inFile.scan_angle_rank <= end)
    if dimension == 'user_data':
        mask = (inFile.user_data >= start) & (inFile.user_data <= end)
    if dimension == 'pt_src_id':
        mask = (inFile.pt_src_id >= start) & (inFile.pt_src_id <= end)
    if dimension == 'gps_time':
        mask = (inFile.gps_time >= start) & (inFile.gps_time <= end)
    if dimension == 'red':
        mask = (inFile.red >= start) & (inFile.red <= end)
    if dimension == 'green':
        mask = (inFile.green >= start) & (inFile.green <= end)
    
    return mask
    # Remove the extension from the file
    #tile_name = tile_name.replace('.las','')
    # Output file
    #outFile = File(tile_name+"_range["+str(start)+":"+str(end)+"].las", mode='w', header=inFile.header)
    # Apply the mask
    #outFile.points = inFile.points[mask]
    # Close the file
    #outFile.close()

# Example run
#if __name__ == "__main__":
#    range('T000.las','x',385652.3738,385700)
