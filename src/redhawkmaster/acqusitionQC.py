import os
from laspy.file import File
import numpy as np
import time
import csv
import datetime
import re

location = './'
header_file_name = 'header.csv'
attribute_qc_file_name = 'attribute.csv'
extra_attr_qc_file_name = 'extra_attributes.csv'

header_exists = os.path.isfile(header_file_name)
attr_exists = os.path.isfile(attribute_qc_file_name)
extra_attr_exists = os.path.isfile(extra_attr_qc_file_name)

header_keys = ['File Name', 'File Signature', 'File Source Id', 'Global Encoding',
               'Gps Time Type', 'Project Id', 'Version Major',
               'Version Minor', 'System Identifier',
               'Generating Software', 'File Creation Date', 'File Creation Day of Year', 'File Creation Year',
               'Header Size', 'Offset to Point Data', 'Number of Variable Length Records',
               'Point Data Format Id', 'Number of point records', 'Number of Points by Return Count',
               'X scale factor', 'Y scale factor', 'Z scale factor',
               'X offset', 'Y offset', 'Z offset',
               'Max X', 'Min X', 'Max Y', 'Min Y', 'Max Z', 'Min Z']

attr_keys = ['File Name', 'Max X', 'Min X', 'Range X', 'Max Y', 'Min Y', 'Range Y', 'Max Z', 'Min Z', 'Range Z',
             'Max Intensity', 'Min Intensity', 'Range Intensity',
             'Max Return Number', 'Min Return Number', 'Range Return Number',
             'Max Number of Returns', 'Min Number of Returns', 'Range Number of Returns',
             'Max Gps Time', 'Min Gps Time', 'Range Gps Time',
             'Max Scan Direction Flag', 'Min Scan Direction Flag', 'Range Scan Direction Flag',
             'Max Edge of Flight Line', 'Min Edge of Flight Line', 'Range Edge of Flight Line',
             'Max Classification', 'Min Classification', 'Range Classification',
             'Max Scan Angle Rank', 'Min Scan Angle Rank', 'Range Scan Angle Rank',
             'Max User Data', 'Min User Data', 'Range User Data',
             'Max Point Source ID', 'Min Point Source ID', 'Range Point Source ID',
             'Max Red', 'Min Red', 'Range Red', 'Max Green', 'Min Green',  'Range Green',
             'Max Blue', 'Min Blue', 'Range Blue']

count = 0

with open(header_file_name, 'a') as header_file, open(attribute_qc_file_name, 'a') as attr_file, \
        open(extra_attr_qc_file_name, 'a') as extra_attr:
    header_csv = csv.writer(header_file)
    attr_csv = csv.writer(attr_file)
    extra_attr_csv = csv.writer(extra_attr)

    header_is_empty = os.stat(header_file_name).st_size == 0
    attr_is_empty = os.stat(attribute_qc_file_name).st_size == 0
    extra_attr_is_empty = os.stat(attribute_qc_file_name).st_size == 0

    extra_dims_keys = ['File Name']
    extra_dims_values = []
    for las_file in os.listdir(location):
        filename, file_extension = os.path.splitext(las_file)
        if file_extension == '.las':
            print(las_file+" starting !!!")
            test = File(las_file, mode='r')

            if header_is_empty:
                header_csv.writerow(header_keys)

            if attr_is_empty:

                attr_csv.writerow(attr_keys)

            date_of_creation = test.header.get_date()

            header_csv.writerow([las_file, test.header.file_signature, test.header.file_source_id,
                                 test.header.global_encoding, test.header.gps_time_type, test.header.guid,
                                 test.header.version_major, test.header.version_minor,
                                 re.sub(r'[\x00]', r'', test.header.system_id), re.sub(r'[\x00]', r'', test.header.software_id),
                                 date_of_creation.strftime('%d-%B-%Y'), date_of_creation.strftime('%j'),
                                 date_of_creation.year, test.header.header_size, test.header.data_offset,
                                 len(test.header.vlrs), test.header.data_format_id, test.header.records_count,
                                 test.header.point_return_count,
                                 test.header.scale[0], test.header.scale[1], test.header.scale[2],
                                 test.header.offset[0], test.header.offset[1], test.header.offset[2],
                                 test.header.max[0], test.header.min[0], test.header.max[1],
                                 test.header.min[1], test.header.min[2], test.header.min[2]])

            maxX = max(test.x)
            minX = min(test.x)
            rangeX = maxX - minX
            maxY = max(test.y)
            minY = min(test.y)
            rangeY = maxY - minY
            maxZ = max(test.z)
            minZ = min(test.z)
            rangeZ = maxZ - minZ
            maxIntensity = max(test.intensity)
            minIntensity = min(test.intensity)
            rangeIntensity = maxIntensity - minIntensity
            maxReturnNumber = max(test.return_num)
            minReturnNumber = min(test.return_num)
            rangeReturnNumber = maxReturnNumber - minReturnNumber
            maxNumberOfReturns = max(test.num_returns)
            minNumberOfReturns = min(test.num_returns)
            rangeNumberOfReturns = maxNumberOfReturns - minNumberOfReturns
            maxGps = max(test.gps_time)
            minGps = min(test.gps_time)
            rangeGps = maxGps - minGps
            maxScanDirectionFlag = max(test.scan_dir_flag)
            minScanDirectionFlag = min(test.scan_dir_flag)
            rangeScanDirectionFlag = maxScanDirectionFlag - minScanDirectionFlag
            maxEdgeOfFlightLine = max(test.edge_flight_line)
            minEdgeOfFlightLine = min(test.edge_flight_line)
            rangeEdgeOfFlightLine = maxEdgeOfFlightLine - minEdgeOfFlightLine
            maxClassification = max(test.classification)
            minClassification = min(test.classification)
            rangeClassification = maxClassification - minClassification
            maxScanAngleRank = max(test.scan_angle_rank)
            minScanAngleRank = min(test.scan_angle_rank)
            rangeScanAngleRank = maxScanAngleRank - int(minScanAngleRank)
            maxUserData = max(test.user_data)
            minUserData = min(test.user_data)
            rangeUserData = maxUserData - minUserData
            maxPointSourceId = max(test.pt_src_id)
            minPointSourceId = min(test.pt_src_id)
            rangePointSourceId = maxPointSourceId - minPointSourceId
            maxRed = max(test.red)
            minRed = min(test.red)
            rangeRed = maxRed - minRed
            maxGreen = max(test.green)
            minGreen = min(test.green)
            rangeGreen = maxGreen - minGreen
            maxBlue = max(test.blue)
            minBlue = min(test.blue)
            rangeBlue = maxBlue - minBlue
            attr_values = [las_file,
                           maxX, minX, rangeX, maxY, minY, rangeY, maxZ, minZ, rangeZ,
                           maxIntensity, minIntensity, rangeIntensity,
                           maxReturnNumber, minReturnNumber, rangeReturnNumber,
                           maxNumberOfReturns, minNumberOfReturns, rangeNumberOfReturns,
                           maxGps, minGps, rangeGps,
                           maxScanDirectionFlag, minScanDirectionFlag, rangeScanDirectionFlag,
                           maxEdgeOfFlightLine, minEdgeOfFlightLine, rangeEdgeOfFlightLine,
                           maxClassification, minClassification, rangeClassification,
                           maxScanAngleRank, minScanAngleRank, rangeScanAngleRank,
                           maxUserData, minUserData, rangeUserData,
                           maxPointSourceId, minPointSourceId, rangeUserData,
                           maxRed, minRed, rangeRed, maxGreen, minGreen, rangeGreen, maxBlue, minBlue, rangeBlue]

            attr_csv.writerow(attr_values)
            extra_dims_values_row = [filename]

            extra_dims = False
            has_extra_dims = False
            for key, val in test.point_format.lookup.items():
                if extra_dims:
                    has_extra_dims = True
                    break
                if key == 'blue':
                    extra_dims = True

            extra_dims = False
            if has_extra_dims:
                for key, val in test.point_format.lookup.items():
                    if extra_dims:

                        maxDim = max(test.reader.get_dimension(key))
                        minDim = min(test.reader.get_dimension(key))
                        rangeDim = maxDim - int(minDim)

                        if ("Max "+key) not in extra_dims_keys:
                            extra_dims_keys.append('Max '+key)

                        if ("Min " + key) not in extra_dims_keys:
                            extra_dims_keys.append('Min ' + key)

                        if ("Range " + key) not in extra_dims_keys:
                            extra_dims_keys.append('Range ' + key)

                        extra_dims_values_row.append(maxDim)
                        extra_dims_values_row.append(minDim)
                        extra_dims_values_row.append(rangeDim)

                    if key == 'blue':
                        extra_dims = True
            else:
                extra_dims_values_row.append('')

            extra_dims_values.append(extra_dims_values_row)

            extra_dims = False

            if count == 5:
                break
            count += 1
            test.close()

    if extra_attr_is_empty:
        extra_attr_csv.writerow(extra_dims_keys)

    for line in extra_dims_values:
        extra_attr_csv.writerow(line)

