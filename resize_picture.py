import cv2
import sys

inputImage = cv2.imread(sys.argv[1])

new_image = cv2.resize(inputImage, (int(inputImage.shape[1]*2), int(inputImage.shape[0]*2)), interpolation=cv2.INTER_AREA)

cv2.imwrite(sys.argv[1], new_image)
