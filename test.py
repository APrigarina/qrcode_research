import cv2
import numpy as np
import sys
import time
from itertools import islice
import optparse
import os
import sys
from os.path import join

if len(sys.argv)>1:
    inputImage = cv2.imread(sys.argv[1])
else:
   inputImage = cv2.imread("build/version_5_right.jpg")

print(inputImage.shape)
showImage = inputImage.copy()

def display(im, bbox):
   n = len(bbox)
   print(n)
   print(bbox.shape)
   for j in range(n):
       cv2.putText(im, str(j), tuple(bbox[j]), 1, 1, (255,0,0), 2)
       cv2.line(im, tuple(bbox[j]), tuple(bbox[ (j+1) % n]), (255,0,0), 2)


qrDecoder = cv2.QRCodeDetector()


ok, bbox = qrDecoder.detect(inputImage)

print(bbox)
# print(inputImage.shape)
if bbox is None:
    print("QR Code not detected")

else:

    data, rectifiedImage = qrDecoder.decode(inputImage, bbox)
    display(showImage, bbox[0])
    # print(cv2.__file__)


    if len(data)>0:

        rectifiedImage = np.uint8(rectifiedImage)
        # cv2.imwrite("/home/annaprigarina/Documents/experiments/new.jpg", rectifiedImage)
        # cv2.namedWindow('Rectified QRCode', cv2.WINDOW_NORMAL)
        # cv2.imshow("Rectified QRCode", rectifiedImage)
        # cv2.waitKey(0)

        print("Decoded Data : {}".format(data))

    else:
        print("QR Code not decoded")

    # cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    # cv2.imshow("Results", showImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
