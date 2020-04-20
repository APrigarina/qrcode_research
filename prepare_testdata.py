import cv2 
import numpy as np
import matplotlib.pyplot as plt
import sys
import optparse
import os
import sys
from os.path import join
import csv
import json

p = optparse.OptionParser()
p.add_option('--input', '-i', default="build/qrcode-datasets/datasets",help="Location of directory with input images")
p.add_option('--data', '-d', default="build/qrcode-datasets/bitmaps",help="Location of directory with decoded data")
p.add_option('--multiple', '-m')

options, arguments = p.parse_args()

dir_images = options.input
dir_decoded_data = options.data
is_multiple = options.multiple

input_path = dir_images

input_dir = [f for f in sorted(os.listdir(input_path))]
for f in os.listdir(dir_decoded_data):
    if f.endswith(".csv"):
        decoded_file = f
path_to_decoded_data = os.path.join(dir_decoded_data, decoded_file)
csv_file_decoded = open(path_to_decoded_data)
csv_reader_decoded = csv.reader(csv_file_decoded)
csv_list_decoded = list(csv_reader_decoded)
csv_dict_decoded = dict(csv_list_decoded)
print(csv_dict_decoded)

result_data = {}

for cat in input_dir:

    print("category", cat)
    images = []
    cat_data = []

    path_to_cat  = os.path.join(input_path, cat)
    input_images = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".png")]
    input_data = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith(".csv")]
    input_points = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith(".txt")]
    # print(input_images)
    for idx, input_image in enumerate(input_images):
        img_dict = {}
        # print(input_image)
        points = []
        data_keys = []
        infos = []
        images.append(input_image)
        img_dict['image_name'] = input_image

        path_to_image = os.path.join(path_to_cat,input_image)
        path_to_data = os.path.join(path_to_cat,input_data[idx])
        path_to_points = os.path.join(path_to_cat,input_points[idx])
        # print(input_points[idx])

        txt_file = open(path_to_points)
        number_qr_codes = 0
        for row in txt_file:
            number_qr_codes += 1
            points.append(row.split())
        # print(number_qr_codes)
        # print(points)
        img_dict['points'] = points

        csv_file = open(path_to_data)
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_list = list(csv_reader)

        keys = [k[-1] for k in csv_list[1:]]
        # print(keys)

        values = [csv_dict_decoded[key] for key in keys]
        img_dict['infos'] = values
        # print(values)
        data_keys.append(keys)
        infos.append(values)

        cat_data.append(img_dict)

    print(cat_data)
    result_data['test_images'] = cat_data
        

    # result_data[cat] = cat_data

    json_object = json.dumps(result_data) 
    name = cat + "_config.json"
    with open(name, "w") as outfile: 
        outfile.write(json_object) 