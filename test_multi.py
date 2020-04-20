import cv2
import numpy as np
import sys
import time
from itertools import islice
from shapely.geometry import Polygon
import optparse
import os
import sys
from os.path import join
from shapely.geometry import Polygon

if len(sys.argv)>1:
    inputImage = cv2.imread(sys.argv[1])
else:
   inputImage = cv2.imread("build/version_5_right.jpg")

print(inputImage.shape)
showImage = inputImage.copy()

def display(im, bbox):
   n = len(bbox)
   print("bbox", n)
   for i in range(n):
       print("i", i)
       print(bbox[i])
       for j in range(4):
           cv2.line(im, tuple(bbox[i][j]), tuple(bbox[i][ (j+1) % 4]), (255,0,0), 3)


qrDecoder = cv2.QRCodeDetector()


ok, bbox = qrDecoder.detectMulti(inputImage)

# print(bbox)
# print(inputImage.shape)
if bbox is None:
    print("QR Code not detected")

else:

    ok, data,rectifiedImage = qrDecoder.decodeMulti(inputImage, bbox)
    print(len(rectifiedImage))
    print(len(data))
    display(showImage, bbox)


    if len(data)>0:
        for i,l in enumerate(data):
    #
    #     # rectifiedImage = np.uint8(rectifiedImage);
    #     # # cv2.imwrite("/home/annaprigarina/Documents/experiments/new.jpg", rectifiedImage)
    #         cv2.namedWindow('Rectified QRCode', cv2.WINDOW_NORMAL)
    #         cv2.imshow("Rectified QRCode", rectifiedImage)
    #         cv2.waitKey(0)
    #         #
            print("Decoded Data : {}".format(l))
    #
    # else:
    #     print("QR Code not decoded")

    cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    cv2.imshow("Results", showImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
