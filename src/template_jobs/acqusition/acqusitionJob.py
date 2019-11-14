import pathmagic
from redhawkmaster.acqusitionQC import extract_qc
import argparse
assert pathmagic

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--location', help='Location of the las file with dash in the end.', required=True)
parser.add_argument('-lqc', '--locationQC', help='Location of the qc las file with dash in the end.',
                    required=True)
parser.add_argument('-attr', '--attribute_name', help='Name of the attribute csv file. Must end with .csv.',
                    default='attribute_name.csv')
parser.add_argument('-head', '--header_name', help='Name of the header csv file. Must end with .csv.',
                    default='header.csv')
parser.add_argument('-exattr', '--extra_attr', help='Name of the extra attribute csv file. Must end with .csv.',
                    default='extra_attr.csv')

args = parser.parse_args()

extract_qc(location=args.location,
           location_qc=args.locationQC,
           attribute_name=args.attribute_name,
           header_name=args.header_name,
           extra_attr_name=args.extra_attr)
