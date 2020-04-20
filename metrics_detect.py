import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import optparse
import os
import sys
from os.path import join
import csv


p = optparse.OptionParser()
p.add_option('--input', '-i', default="build/qrcode-datasets/datasets",help="Location of directory with input images")
p.add_option('--multiple', '-m')

options, arguments = p.parse_args()

dir_images = options.input
is_multiple = options.multiple

def display(im, bbox):
   n = len(bbox)
   for j in range(n):
       cv2.line(im, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255,0,0), 3)

qrDecoder = cv2.QRCodeDetector()

def save_graph(categories, detect_stat, decode_stat, description):
    cat_pos = np.arange(len(categories))
    plt.subplot(2,1,1)
    plt.title("Detection percent " + description, pad=20.0)
    plt.bar(cat_pos, detect_stat, color = (0.5,0.1,0.5,0.6))
    plt.xlabel('categories')
    plt.ylabel('detect_percent')
    plt.ylim(0,100)
    plt.xticks(cat_pos, categories)
    for i, v in enumerate(detect_stat):
        plt.text(x=i-0.15, y=v+1, s=str(v), color=(0,0,0))

    plt.tight_layout()
    plt.subplot(2,1,2)
    plt.title("Decoding percent " + description, pad=20.0)
    plt.bar(cat_pos, decode_stat, color = (0.5,0.1,0.5,0.6))
    plt.xlabel('categories')
    plt.ylabel('decode_percent')
    plt.ylim(0,100)
    plt.xticks(cat_pos, categories)
    for i, v in enumerate(decode_stat):
        plt.text(x=i-0.15, y=v+1, s=str(v), color=(0,0,0))

    plt.show()

def compute_metric_3():

    detection_stat = []
    decoding_stat = []
    categories = []
    not_detected = 0
    detected = 0

    not_decoded = 0
    decoded = 0

    input_path = dir_images

    input_dir = [f for f in sorted(os.listdir(input_path))]
    # print(input_dir)

    for cat in input_dir:

        categories.append(cat)
        print("category", cat)
        print()


        path_to_cat  = os.path.join(input_path, cat)
        input_images = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png")]
        input_data = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("csv")]
        for idx, input_image in enumerate(input_images):
            print(input_image)
            path_to_image = os.path.join(path_to_cat,input_image)
            path_to_data = os.path.join(path_to_cat,input_data[idx])

            number_qr_codes = 0

            with open(path_to_data) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        number_qr_codes = int(row[0][0])
                        break
            print("number qr codes", number_qr_codes)
            image_to_detect = cv2.imread(path_to_image)
            show_image = image_to_detect.copy()
            b, bbox = qrDecoder.detect(image_to_detect)
            if bbox is not None:

                not_detected += (number_qr_codes - 1)
                detected += 1

                # display(show_image, bbox)
                # cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
                # cv2.imshow("Results", show_image)
                # cv2.waitKey(0)

                data, rec = qrDecoder.decode(image_to_detect, bbox)

                if len(data)>0:
                    print("Decoded Data : {}".format(data))
                    decoded += 1
                    not_decoded += (number_qr_codes - 1)
                else:
                    not_decoded += number_qr_codes

            else:
                not_detected += number_qr_codes

        if detected != 0 and not_detected != 0:
            total = detected + not_detected
            print("total", total)
            print("detected", detected)
            print("not detected", not_detected)
            detected_percent = np.round((detected/total) * 100, 2)
            print("detected percent: {}%\n".format(detected_percent))
            detection_stat.append(detected_percent)
            print("decoded", decoded)
            print("not_decoded", not_decoded)
            decoded_percent = np.round((decoded/total) * 100, 2)
            print("decoded percent: {}%\n\n".format(decoded_percent))
            decoding_stat.append(decoded_percent)

    save_graph(categories, detection_stat, decoding_stat, "among all QR-codes")

compute_metric_3()
