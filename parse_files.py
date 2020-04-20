import xml.etree.ElementTree as ET
import os
import sys
import optparse
from os.path import join
import re
import json

p = optparse.OptionParser()
p.add_option('--input', '-i', default="build/qrcode-datasets/datasets",help="Location of directory with input data")

options, arguments = p.parse_args()

dir_data = options.input

for category in sorted(os.listdir(dir_data)):
    input_path = os.path.join(dir_data,category)


    for f in os.listdir(input_path):
        if f.endswith(".xml"):
            path_to_file = os.path.join(input_path, f)
            tree = ET.parse(path_to_file)
            root = tree.getroot()

    for img in root.iter("image"):
        name = img.get("name")
        root, ext = os.path.splitext(name)
        path_to_annotation = os.path.join(input_path, root + '.txt')

        annotation = open(path_to_annotation, "w")        
    
        all_points = []
        for polygon in img.iter("polygon"):
            points = polygon.get("points")
            points = re.split(";|,", points)
            result = ' '.join([str(cvRound(x)) for x in points])
            annotation.write(result + '\n') 

        annotation.close()       
    
