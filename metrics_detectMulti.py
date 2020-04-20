import csv
import optparse
import os
import sys
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

p = optparse.OptionParser()
p.add_option('--input', '-i', default="build/qrcode-datasets/datasets",help="Location of directory with input images")
p.add_option('--multiple', '-m')

options, arguments = p.parse_args()

dir_images = options.input
is_multiple = options.multiple

input_path = dir_images

input_dir = [f for f in sorted(os.listdir(input_path))]

def display(im, bbox):
   n = len(bbox)
   # print(bbox)
   for i in range(n):
       for j in range(4):
           cv2.line(im, tuple(bbox[i][j]), tuple(bbox[i][ (j+1) % 4]), (255,0,0), 3)

qrDecoder = cv2.QRCodeDetector()




def show_detect_decode_bar(categories, detect_stat, decode_stat, description):
    cat_pos = np.arange(len(categories))
    # plt.figure(figsize=(15,15))

    plt.subplot(2,1,1)
    plt.title("Detection percent " + description, pad=20.0)
    plt.bar(cat_pos, detect_stat, color = (0.5,0.1,0.5,0.6))
    plt.xlabel('categories')
    plt.ylabel('detect_percent')
    plt.ylim(0,100)
    plt.xticks(cat_pos, categories)
    for i, v in enumerate(detect_stat):
        plt.text(x=i-0.15, y=v+1, s=str(np.round(v, 2)), color=(0,0,0))

    plt.tight_layout()
    plt.subplot(2,1,2)
    plt.title("Decoding percent " + description, pad=20.0)
    plt.bar(cat_pos, decode_stat, color = (0.5,0.1,0.5,0.6))
    plt.xlabel('categories')
    plt.ylabel('decode_percent')
    plt.ylim(0,100)
    plt.xticks(cat_pos, categories)
    for i, v in enumerate(decode_stat):
        plt.text(x=i-0.15, y=v+1, s=str(np.round(v, 2)), color=(0,0,0))

    plt.show()

def show_multiple_bar(categories, detect_stat1, decode_stat1, detect_stat2, decode_stat2, description1, description2):
    cat_pos = np.arange(len(categories))
    # plt.figure(figsize=(20,20))

    plt.subplot(2,1,1)
    plt.title("Detection percent" , pad=20.0)

    plt.bar(cat_pos, detect_stat2, width=0.6, color=(0.56, 0.68, 0.84), label=description2)
    plt.bar(cat_pos-0.2, detect_stat1, width=0.6, color=(0.29, 0.52, 0.8), label=description1)

    plt.xlabel('categories')
    plt.ylabel('detect_percent')
    plt.ylim(0,105)
    plt.xticks(cat_pos, categories)
    for i, v in enumerate(detect_stat1):
        plt.text(x=i-0.3, y=v-2, s=str(np.round(v, 2)), color=(0,0,0))
    for i, v in enumerate(detect_stat2):
        plt.text(x=i-0.1, y=v+1, s=str(np.round(v, 2)), color=(0,0,0))

    plt.legend(loc='best')

    plt.tight_layout()
    plt.subplot(2,1,2)
    plt.title("Decoding percent" , pad=20.0)

    plt.bar(cat_pos, decode_stat2, width=0.6, color=(0.56, 0.68, 0.84), label=description2)
    plt.bar(cat_pos-0.2, decode_stat1, width=0.6, color=(0.29, 0.52, 0.8), label=description1)

    plt.xlabel('categories')
    plt.ylabel('decode_percent')
    plt.xticks(cat_pos, categories)
    plt.ylim(0,105)
    for i, v in enumerate(decode_stat1):
        plt.text(x=i-0.3, y=v-2, s=str(np.round(v, 2)), color=(0,0,0))
    for i, v in enumerate(decode_stat2):
        plt.text(x=i-0.1, y=v+1, s=str(np.round(v, 2)), color=(0,0,0))
    
    plt.legend(loc='best')

    plt.show()
    # plt.savefig('metric1_2.png')

def show_detection_score(categories, f_score):
    cat_pos = np.arange(len(categories))

    plt.title("Detection f-score", pad=20.0)
    plt.bar(cat_pos, f_score, color = (0.5,0.1,0.5,0.6))
    plt.xlabel('categories')
    plt.ylabel('f_score')
    plt.ylim(0,1)
    plt.xticks(cat_pos, categories)
    for i, v in enumerate(f_score):
        plt.text(x=i-0.15, y=v+0.02, s=str(v), color=(0,0,0))

    plt.show()




def iou_graph(categories, iou_stat):
    cat_pos = np.arange(len(categories))
    plt.title("Intersection over Union", pad=20.0)
    plt.bar(cat_pos, iou_stat, color = (0.3,0.1,0.6,0.6))
    plt.xlabel('categories')
    plt.ylabel('IoU')
    # plt.ylim(0,1)
    plt.xticks(cat_pos, categories)
    for i, v in enumerate(iou_stat):
        plt.text(x=i-0.2, y=v+0.02, s=str(v), color=(0,0,0))

    plt.show()

def compute_metric_3():

    detection_data = {}
    decoding_data = {}
    detection_stat = []
    decoding_stat = []
    categories = []



    for cat in input_dir:
        if cat != 'more_medium':
            
            continue
        detected = 0
        decoded = 0
        
        categories.append(cat)
        print("category", cat)
        print()
        
        number_of_all_qr_codes = 0


        path_to_cat  = os.path.join(input_path, cat)
        input_images = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png")]
        input_data = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("csv")]
        # print(input_images)


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
                        number_qr_codes = int(row[0].split()[0])
                        break
            print("number qr codes", number_qr_codes)
            number_of_all_qr_codes += number_qr_codes
            print("number_of_all_qr_codes", number_of_all_qr_codes)
            # print(input_image)
            image_to_detect = cv2.imread(path_to_image)
            show_image = image_to_detect.copy()
            # print(image_to_detect.shape)
            b, bbox = qrDecoder.detectMulti(image_to_detect)
            if bbox is not None:
                print("bbox", len(bbox))
                print(bbox)

                # not_detected += (number_qr_codes - len(bbox))
                detected += len(bbox)

                # display(show_image, bbox)
                # cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
                # cv2.imshow("Results", show_image)
                # cv2.waitKey(0)

                ok, data, rec = qrDecoder.decodeMulti(image_to_detect, bbox)
                #
                print("data", len(data))
                if len(data)>0:
                    for i, l in enumerate(data):
                        if len(l) > 0:
                            # print("decoded size", rec[i].shape)
                            print("Decoded Data : {}".format(l))
                            decoded += 1



        total = number_of_all_qr_codes
        print("total", total)
        print("detected", detected)
        # print("not detected", not_detected)
        detected_percent = np.round((detected/total) * 100, 5)
        print("detected percent: {}%\n".format(detected_percent))
        detection_stat.append(detected_percent)
        print("decoded", decoded)
        # print("not_decoded", not_decoded)
        decoded_percent = np.round((decoded/total) * 100, 5)
        print("decoded percent: {}%\n\n".format(decoded_percent))
        decoding_stat.append(decoded_percent)
        
        detection_data[cat] = detected_percent
        decoding_data[cat] = decoded_percent

    print(detection_data)
    print(decoding_data)
    json_detection = json.dumps(detection_data) 
    with open("detection_percent_metric3.json", "w") as outfile: 
        outfile.write(json_detection) 
    json_decoding = json.dumps(decoding_data) 
    with open("decoding_percent_metric3.json", "w") as outfile: 
        outfile.write(json_decoding) 

    show_detect_decode_bar(categories, detection_stat, decoding_stat, "among all QR-codes")

def compute_metric_2():

    detection_data = {}
    decoding_data = {}
    detection_stat = []
    decoding_stat = []
    categories = []


    for cat in input_dir:
        
        detected = 0
        not_detected = 0
        decoded = 0
        not_decoded = 0



        categories.append(cat)
        print("category", cat)
        print()


        path_to_cat  = os.path.join(input_path, cat)
        input_images = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png")]
        input_data = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("csv")]
        # print(input_images)
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
                        number_qr_codes = int(row[0].split()[0])
                        break
            print("number qr codes", number_qr_codes)
            # print(input_image)
            image_to_detect = cv2.imread(path_to_image)
            show_image = image_to_detect.copy()

            b, bbox = qrDecoder.detectMulti(image_to_detect)
            print(bbox)
            if bbox is not None:
                print(bbox.shape)
                detected += 1
                ok, data, rec = qrDecoder.decodeMulti(image_to_detect, bbox)
                print("data", len(data))
                if len(data)>0:
                    flag = False
                    for l in data:
                        print(len(l))
                        if len(l) > 0:
                            flag = True
                    if flag:    
                        decoded += 1
                else:
                    not_decoded += 1
            else:
                not_detected += 1






        total = len(input_images)
        print("total number of images", total)
        print("detected", detected)
        print("not detected", not_detected)
        detected_percent = np.round((detected/total) * 100, 5)
        print("detected image percent: {}%\n".format(detected_percent))
        detection_stat.append(detected_percent)
        print("decoded", decoded)
        print("not_decoded", not_decoded)
        decoded_percent = np.round((decoded/total) * 100, 5)
        print("decoded percent: {}%\n\n".format(decoded_percent))
        decoding_stat.append(decoded_percent)

        detection_data[cat] = detected_percent
        decoding_data[cat] = decoded_percent

    json_detection = json.dumps(detection_data) 
    with open("detection_percent_metric2.json", "w") as outfile: 
        outfile.write(json_detection) 
    json_decoding = json.dumps(decoding_data) 
    with open("decoding_percent_metric2.json", "w") as outfile: 
        outfile.write(json_decoding) 
    # detection_stat1, decoding_stat1 = compute_metric_1()
    return detection_stat, decoding_stat

    # show_detect_decode_bar(categories, detection_stat, decoding_stat, "of pictures where at least 1 QR-code is detected/decoded")


def compute_metric_1():

    detection_data = {}
    decoding_data = {}
    detection_stat = []
    decoding_stat = []
    categories = []

    for cat in input_dir:

        detected = 0
        not_detected = 0
        decoded = 0
        not_decoded = 0

        # if cat != 'more_medium':
        #     continue
        categories.append(cat)
        print("category", cat)


        path_to_cat  = os.path.join(input_path, cat)
        input_images = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png")]
        input_data = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("csv")]
        # print(input_images)
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
                        number_qr_codes = int(row[0].split()[0])
                        break
            print("number qr codes", number_qr_codes)
            # print(input_image)
            image_to_detect = cv2.imread(path_to_image)
            show_image = image_to_detect.copy()
            # print(image_to_detect.shape)
            b, bbox = qrDecoder.detectMulti(image_to_detect)
            print(bbox)
            if bbox is not None:
                print("bbox", bbox.shape)
                # print(len(bbox))
                if number_qr_codes == len(bbox):
                    detected += 1
                else:
                    not_detected += 1
                ok, data, rec = qrDecoder.decodeMulti(image_to_detect, bbox)


                if len(data)>0:
                    print("data", len(data))
                    number_decoded = 0
                    for i, l in enumerate(data):
                        print("data i", i, len(l))
                        if len(l) > 0:
                            number_decoded += 1
                    if number_decoded == number_qr_codes:
                        decoded +=1
                    else:
                        not_decoded += 1

            else:
                not_detected += 1





        total = len(input_images)
        print("total", total)
        print("detected", detected)
        print("not_detected", not_detected)
        detected_percent = np.round((detected/total) * 100, 5)
        print("detected percent: {}%\n".format(detected_percent))
        detection_stat.append(detected_percent)
        print("decoded", decoded)
        print("not_decoded", not_decoded)
        decoded_percent = np.round((decoded/total) * 100,5)
        print("decoded percent: {}%\n\n".format(decoded_percent))
        decoding_stat.append(decoded_percent)
        detection_data[cat] = detected_percent
        decoding_data[cat] = decoded_percent

    json_detection = json.dumps(detection_data) 
    with open("detection_percent_metric1.json", "w") as outfile: 
        outfile.write(json_detection) 
    json_decoding = json.dumps(decoding_data) 
    with open("decoding_percent_metric1.json", "w") as outfile: 
        outfile.write(json_decoding) 

    return detection_stat, decoding_stat
    # show_detect_decode_bar(categories, detection_stat, decoding_stat, "of pictures where all QR-codes are detected/decoded")

def list_to_points(points):
    result = []
    for i in range(0, len(points), 2):
        result.append((points[i], points[i+1]))
    return result

def intersection(a1, a2, b1, b2):
    c1 = ((a1[0] * a2[1]  -  a1[1] * a2[0]) * (b1[0] - b2[0]) - (b1[0] * b2[1]  -  b1[1] * b2[0]) * (a1[0] - a2[0])) / ((a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0]))

    c2 = ((a1[0] * a2[1]  -  a1[1] * a2[0]) * (b1[1] - b2[1]) - (b1[0] * b2[1]  -  b1[1] * b2[0]) * (a1[1] - a2[1])) / ((a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0]))

    return np.array([c1, c2])

def get_data_centers(points):
    data_centers = []
    for p in points:
        center = intersection(p[0], p[2], p[1], p[3])
        data_centers.append(center)
    return data_centers

def get_min_dist(bbox_centers, true_points_centers):
    nearest_centers = []
    for bb_center in bbox_centers:
        nearest_center = []
        for tp_center in true_points_centers:
            dist = cv2.norm(bb_center - tp_center)
            nearest_center.append(dist)
        
        min_idx = np.argmin(nearest_center)
        min_value = min(nearest_center)
        nearest_centers.append((min_idx, min_value))
    # print("nearest centers ", nearest_centers)
    return nearest_centers          

def compute_metric_4():

    iou_data = {}
    inter_over_union = []
    categories = []

    input_path = dir_images

    input_dir = [f for f in sorted(os.listdir(input_path))]
    # print(input_dir)

    to_change = {}

    for cat in input_dir:

        img_stat = []
        img_to_change = []

        categories.append(cat)
        if cat != 'more_medium':
            continue
        print("category", cat)

        path_to_cat  = os.path.join(input_path, cat)
        input_images = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png")]
        input_data = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("txt")]
        # print(input_images)
        for idx, input_image in enumerate(input_images):

            if input_image != "IMG_2763.JPG":
                continue
            print(input_image)
            path_to_image = os.path.join(path_to_cat,input_image)
            path_to_data = os.path.join(path_to_cat,input_data[idx])

            number_qr_codes = 0

            # with open(path_to_data) as csv_file:
            #     csv_reader = csv.reader(csv_file, delimiter=',')
            #     line_count = 0
            #     for row in csv_reader:
            #         if line_count == 0:
            #             number_qr_codes = int(row[0][0])
            #             break
            annotation = open(path_to_data, 'r')
            true_points = []
            for line in annotation:
                if len(line) > 0:
                    number_qr_codes += 1
                    temp_points = [float(x) for x in line.split()]
                    true_points.append(list_to_points(temp_points))
            # print("number qr codes", number_qr_codes)
            # print(true_points)

            image_to_detect = cv2.imread(path_to_image)
            size = image_to_detect.shape[:2]
            show_image = image_to_detect.copy()
            # print(image_to_detect.shape)
            ok, bboxes = qrDecoder.detectMulti(image_to_detect)

            true_points_centers = get_data_centers(true_points)
            # print("true_points_centers", true_points_centers)

            if bboxes is not None:
                bbox_stat = 0

                bbox_centers = get_data_centers(bboxes)
                # print("bbox_centers", bbox_centers)

                bbox_nearest_centers = get_min_dist(bbox_centers, true_points_centers)
                # print(cv2.__file__)
                # cv2.namedWindow('detect_mat', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('true_mat', cv2.WINDOW_NORMAL)
                # cv2.namedWindow("wrong", cv2.WINDOW_NORMAL)

                for idx, point in enumerate(bbox_nearest_centers):
                    idx_bbox = idx
                    idx_true_bbox = point[0]

                    detected_bbox = bboxes[idx]
                    matched_true_bbox = true_points[idx_true_bbox]

                    # diff = detected_bbox - matched_true_bbox
                    

                    # if (np.max(diff) > 10):
                    #     print(detected_bbox)
                    #     print(matched_true_bbox)
                    #     cv2.circle(image_to_detect, (int(matched_true_bbox[0][0]), int(matched_true_bbox[0][1])), 15, (0, 255, 0), -1)
                    #     cv2.circle(image_to_detect, (int(matched_true_bbox[1][0]), int(matched_true_bbox[1][1])), 15, (0, 0, 255), -1)
                    #     cv2.circle(image_to_detect, (int(matched_true_bbox[2][0]), int(matched_true_bbox[2][1])), 15, (255), -1)
                    #     cv2.circle(image_to_detect, (int(matched_true_bbox[3][0]), int(matched_true_bbox[3][1])), 15, (0,0,0), -1)
                    #     cv2.imshow("wrong", image_to_detect)
                    #     cv2.waitKey(0)
                    #     # print(diff)
                    #     # print(np.max(diff))
                    #     img_to_change.append(input_image)
                    

                    # break

                    detect_mask = np.zeros(size, np.uint8)
                    true_mask = np.zeros(size, np.uint8)

                    cv2.fillPoly(detect_mask, np.array([detected_bbox], dtype=np.int32), (255))
                    # cv2.imshow("detect_mat", detect_mask)
                    # cv2.waitKey(0)
                    
                    cv2.fillPoly(true_mask, np.array([matched_true_bbox], dtype=np.int32), (255))
                    # cv2.imshow("true_mat", true_mask)
                    # cv2.waitKey(0)
 
                    union = detect_mask | true_mask
                    n_white_pix_union = np.sum(union == 255)

                    inter = detect_mask & true_mask
                    n_white_pix_inter = np.sum(inter == 255)
                    print("intersection ", n_white_pix_inter)
                    print("union ", n_white_pix_union)
                    print("intersection over union = ", n_white_pix_inter / n_white_pix_union)

                    iou = n_white_pix_inter / n_white_pix_union

                    bbox_stat += iou
                
                bbox_stat /= len(bboxes)

                img_stat.append(bbox_stat)    

            
        cat_stat = np.round(np.mean(img_stat),5)
        inter_over_union.append(cat_stat)
        iou_data[cat] = cat_stat
        # to_change[cat] = img_to_change
    json_iou = json.dumps(iou_data) 
    with open("detection_percent_metrci4.json", "w") as outfile: 
        outfile.write(json_iou)

    # print(to_change)
    print("for all categories", inter_over_union)
    # iou_graph(categories, inter_over_union)



def compute_metric_5():

    fscore_data = {}
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    f_score_list = []

    categories = []

    for cat in input_dir:

        tp = 0
        fp = 0
        fn = 0

        categories.append(cat)
        # if cat != 'img_1080p':
        #     continue
        print("category", cat)


        path_to_cat  = os.path.join(input_path, cat)
        input_images = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png")]
        input_data = [f for f in sorted(os.listdir(path_to_cat)) if f.endswith("txt")]
        # print(input_images)
        for idx, input_image in enumerate(input_images):

            print(input_image)
            path_to_image = os.path.join(path_to_cat,input_image)
            path_to_data = os.path.join(path_to_cat,input_data[idx])

            number_qr_codes = 0

            annotation = open(path_to_data, 'r')
            true_points = []
            for line in annotation:
                if len(line) > 0:
                    number_qr_codes += 1
                    temp_points = [float(x) for x in line.split()]
                    true_points.append(list_to_points(temp_points))
            print("number qr codes", number_qr_codes)
            # print(true_points)

            image_to_detect = cv2.imread(path_to_image)
            size = image_to_detect.shape[:2]
            show_image = image_to_detect.copy()
            # print(image_to_detect.shape)
            ok, bboxes = qrDecoder.detectMulti(image_to_detect)

            true_points_centers = get_data_centers(true_points)
            # print("true_points_centers", true_points_centers)

            if bboxes is None:
                bboxes = []

            if len(bboxes) > 0:
                bbox_stat = 0

                bbox_centers = get_data_centers(bboxes)
                # print("bbox_centers", bbox_centers)

                bbox_nearest_centers = get_min_dist(bbox_centers, true_points_centers)
                # print(cv2.__file__)
                # cv2.namedWindow('detect_mat', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('true_mat', cv2.WINDOW_NORMAL)
                for idx, point in enumerate(bbox_nearest_centers):
                    idx_bbox = idx
                    idx_true_bbox = point[0]

                    detected_bbox = bboxes[idx]
                    # print(detected_bbox)
                    matched_true_bbox = true_points[idx_true_bbox]
                    # print(matched_true_bbox)



                    detect_mask = np.zeros(size, np.uint8)
                    true_mask = np.zeros(size, np.uint8)

                    cv2.fillPoly(detect_mask, np.array([detected_bbox], dtype=np.int32), (255))
                    # cv2.imshow("detect_mat", detect_mask)
                    # cv2.waitKey(0)
                    
                    cv2.fillPoly(true_mask, np.array([matched_true_bbox], dtype=np.int32), (255))
                    # cv2.imshow("true_mat", true_mask)
                    # cv2.waitKey(0)

                    inter = detect_mask & true_mask
                    n_white_pix_inter = np.sum(inter == 255)

                    if n_white_pix_inter > 0:
                        tp += 1

            print(len(bboxes))
            print(bboxes)
            
            if len(bboxes) > number_qr_codes:
                fp += (len(bboxes) - number_qr_codes)
            elif len(bboxes) < number_qr_codes:
                fn += (number_qr_codes - len(bboxes))

            print("tp={} fp={} fn={}\n".format(tp, fp, fn))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * precision * recall / (precision + recall)
        print("f_score", f_score)
        f_score_list.append(np.round(f_score, 2))

        fscore_data[cat] = np.round(f_score, 5)

    # show_detection_score(categories, f_score_list)
    
    json_fscore = json.dumps(fscore_data) 
    with open("detection_percent_metric5.json", "w") as outfile: 
        outfile.write(json_fscore)

    return f_score_list



# compute_metric_5()
compute_metric_4()
# compute_metric_3()
# compute_metric_2()
# compute_metric_1()

# detection_stat1, decoding_stat1 = compute_metric_1()
# detection_stat2, decoding_stat2 = compute_metric_2()

# show_multiple_bar(input_dir, detection_stat1, decoding_stat1, detection_stat2, decoding_stat2, 'all QR-codes', 'at least 1 QR-code')
