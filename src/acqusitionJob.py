from redhawkmaster.acqusitionQC import extract_qc

extract_qc(location='./',
           location_qc='./qc_dir/',
           attribute_name='attribute.csv',
           header_name='header.csv',
           extra_attr_name='extra_attributes.csv')
