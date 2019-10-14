import os
from laspy.file import File
import numpy as np
import time
import csv
import datetime
import re
from collections import defaultdict
import bisect
from datetime import datetime, timedelta
from gnsscal import date2gpswd

# https://stackoverflow.com/questions/33415475/how-to-get-current-date-and-time-from-gps-unsegment-time-in-python
_LEAP_DATES = ((1981, 6, 30), (1982, 6, 30), (1983, 6, 30),
               (1985, 6, 30), (1987, 12, 31), (1989, 12, 31),
               (1990, 12, 31), (1992, 6, 30), (1993, 6, 30),
               (1994, 6, 30), (1995, 12, 31), (1997, 6, 30),
               (1998, 12, 31), (2005, 12, 31), (2008, 12, 31),
               (2012, 6, 30), (2015, 6, 30), (2016, 12, 31))

LEAP_DATES = tuple(datetime(i[0], i[1], i[2], 23, 59, 59) for i in _LEAP_DATES)


def leap(date):
    """
    Return the number of leap seconds since 1980-01-01

    :param date: datetime instance
    :return: leap seconds for the date (int)
    """
    # bisect.bisect returns the index `date` would have to be
    # inserted to keep `LEAP_DATES` sorted, so is the number of
    # values in `LEAP_DATES` that are less than `date`, or the
    # number of leap seconds.
    return bisect.bisect(LEAP_DATES, date)


def gps2utc(week, secs):
    """
    :param week: GPS week number, i.e. 1866
    :param secs: number of seconds since the beginning of `week`
    :return: datetime instance with UTC time
    """
    secs_in_week = 604800
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    date_before_leaps = gps_epoch + timedelta(seconds=week * secs_in_week + secs)
    return date_before_leaps - timedelta(seconds=leap(date_before_leaps))


def attribute(test, las_file, csv_name='attribute.csv'):
    attr_keys = ['File Name', 'Max X', 'Min X', 'Range X', 'Max Y', 'Min Y', 'Range Y', 'Max Z', 'Min Z', 'Range Z',
                 'Max Intensity', 'Min Intensity', 'Range Intensity',
                 'Max Return Number', 'Min Return Number', 'Range Return Number',
                 'Max Number of Returns', 'Min Number of Returns', 'Range Number of Returns',
                 'Max Gps Time', 'Min Gps Time', 'Range Gps Time',
                 'Max Date Time', 'Min Date Time', 'Range Date Time',
                 'Max Scan Direction Flag', 'Min Scan Direction Flag', 'Range Scan Direction Flag',
                 'Max Edge of Flight Line', 'Min Edge of Flight Line', 'Range Edge of Flight Line',
                 'Max Classification', 'Min Classification', 'Range Classification',
                 'Max Scan Angle Rank', 'Min Scan Angle Rank', 'Range Scan Angle Rank',
                 'Max User Data', 'Min User Data', 'Range User Data',
                 'Max Point Source ID', 'Min Point Source ID', 'Range Point Source ID',
                 'Max Red', 'Min Red', 'Range Red', 'Max Green', 'Min Green',  'Range Green',
                 'Max Blue', 'Min Blue', 'Range Blue']

    attribute_qc_file_name = csv_name

    with open(attribute_qc_file_name, 'a') as attr_file:
        attr_csv = csv.writer(attr_file)

        attr_is_empty = os.stat(attribute_qc_file_name).st_size == 0

        if attr_is_empty:
            attr_csv.writerow(attr_keys)

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

        week_day_of_creation = date2gpswd(test.header.get_date().date())[0]
        maxDayCreation = gps2utc(week_day_of_creation, maxGps)
        minDayCreation = gps2utc(week_day_of_creation, minGps)
        rangeDayCreation = maxDayCreation - minDayCreation

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
        attr_values = [las_file, maxX, minX, rangeX, maxY, minY, rangeY, maxZ, minZ, rangeZ,
                       maxIntensity, minIntensity, rangeIntensity,
                       maxReturnNumber, minReturnNumber, rangeReturnNumber,
                       maxNumberOfReturns, minNumberOfReturns, rangeNumberOfReturns,
                       maxGps, minGps, rangeGps,
                       maxDayCreation.strftime('%d-%B-%Y, %H:%M:%S'), minDayCreation.strftime('%d-%B-%Y, %H:%M:%S'),
                       str(rangeDayCreation),
                       maxScanDirectionFlag, minScanDirectionFlag, rangeScanDirectionFlag,
                       maxEdgeOfFlightLine, minEdgeOfFlightLine, rangeEdgeOfFlightLine,
                       maxClassification, minClassification, rangeClassification,
                       maxScanAngleRank, minScanAngleRank, rangeScanAngleRank,
                       maxUserData, minUserData, rangeUserData,
                       maxPointSourceId, minPointSourceId, rangePointSourceId,
                       maxRed, minRed, rangeRed, maxGreen, minGreen, rangeGreen, maxBlue, minBlue, rangeBlue]

        attr_csv.writerow(attr_values)


def header(test, las_file, csv_name='header.csv'):
    header_keys = ['File Name', 'File Signature', 'File Source Id', 'Global Encoding',
                   'Gps Time Type', 'Project Id', 'Version Major',
                   'Version Minor', 'System Identifier',
                   'Generating Software', 'File Creation Date', 'File Creation Day of Year', 'File Creation Year',
                   'Header Size', 'Offset to Point Data', 'Number of Variable Length Records',
                   'Point Data Format Id', 'Number of point records', 'Number of Points by Return Count',
                   'X scale factor', 'Y scale factor', 'Z scale factor',
                   'X offset', 'Y offset', 'Z offset',
                   'Max X', 'Min X', 'Range X', 'Max Y', 'Min Y', 'Range Y', 'Max Z', 'Min Z', 'Range Z']

    header_file_name = csv_name

    with open(header_file_name, 'a') as header_file:
        header_csv = csv.writer(header_file)

        header_is_empty = os.stat(header_file_name).st_size == 0

        if header_is_empty:
            header_csv.writerow(header_keys)

        date_of_creation = test.header.get_date()

        header_csv.writerow([las_file, test.header.file_signature, test.header.file_source_id,
                             test.header.global_encoding, test.header.gps_time_type, test.header.guid,
                             test.header.version_major, test.header.version_minor,
                             re.sub(r'[\x00]', r'', test.header.system_id),
                             re.sub(r'[\x00]', r'', test.header.software_id),
                             date_of_creation.strftime('%d-%B-%Y'), date_of_creation.strftime('%j'),
                             date_of_creation.year, test.header.header_size, test.header.data_offset,
                             len(test.header.vlrs), test.header.data_format_id, test.header.records_count,
                             test.header.point_return_count,
                             test.header.scale[0], test.header.scale[1], test.header.scale[2],
                             test.header.offset[0], test.header.offset[1], test.header.offset[2],
                             test.header.max[0], test.header.min[0], (test.header.max[0] - test.header.min[0]),
                             test.header.max[1], test.header.min[1], (test.header.max[1] - test.header.min[1]),
                             test.header.max[2], test.header.min[2], (test.header.max[2] - test.header.min[2])])


extra_dims_dict = defaultdict(list)


def extra_attr(test, las_file):
    extra_dims_dict['File Name'].append(las_file)

    extra_dims = False
    has_extra_dims = False
    for key, val in test.point_format.lookup.items():
        if extra_dims:
            has_extra_dims = True
            break
        if key == 'blue':
            extra_dims = True

    extra_dims = False
    keys = [i for i in extra_dims_dict if extra_dims_dict[i] != extra_dims_dict.default_factory()]

    if has_extra_dims:
        extra_dims = False
        has_extra_dims = False
        for key, val in test.point_format.lookup.items():
            if extra_dims:
                if extra_dims_dict['File Name'].index(las_file) != len(extra_dims_dict['Max '+key]):
                    for i in range(0, extra_dims_dict['File Name'].index(las_file)):
                        extra_dims_dict['Max ' + key].insert(0, '')
                        extra_dims_dict['Min ' + key].insert(0, '')
                        extra_dims_dict['Range ' + key].insert(0, '')
                extra_dims_dict['Max '+key].append(max(test.reader.get_dimension(key)))
                extra_dims_dict['Min ' + key].append(min(test.reader.get_dimension(key)))
                extra_dims_dict['Range ' + key].\
                    append(max(test.reader.get_dimension(key)) - min(test.reader.get_dimension(key)))
            if key == 'blue':
                extra_dims = True

        extra_dims = False

    limit = len(extra_dims_dict['File Name'])

    for key in keys:
        column = extra_dims_dict[key]

        if len(column) != limit:
            for i in range(len(column), limit):
                column.append('')


def write_extra_dim(csv_name='extra_attributes.csv'):

    field_names = [i for i in extra_dims_dict if extra_dims_dict[i] != extra_dims_dict.default_factory()]
    with open(csv_name, 'w', ) as header_csv:
        writer = csv.writer(header_csv)

        writer.writerow(field_names)
        writer.writerows(zip(*[extra_dims_dict[key] for key in field_names]))


def extract_qc(location="./", location_qc="./", attribute_name='attribute.csv', header_name='header.csv',
               extra_attr_name='extra_attributes.csv'):

    if not os.path.exists(location_qc):
        os.makedirs(location_qc)

    for las_file in os.listdir(location):
        filename, file_extension = os.path.splitext(las_file)
        if file_extension == '.las':
            print('Starting file: '+filename)
            test = File(las_file, mode='r')
            attribute(test, filename, csv_name=(location_qc+attribute_name))
            header(test, filename, csv_name=(location_qc+header_name))
            extra_attr(test, filename)
            test.close()

    write_extra_dim(csv_name=(location_qc+extra_attr_name))

