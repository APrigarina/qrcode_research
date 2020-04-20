import numpy as np
import cv2

cap = cv2.VideoCapture(0)
qrDecoder = cv2.QRCodeDetector()
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

while(cv2.waitKey(1) < 0):

    ret, frame = cap.read()
    # print(frame.shape)
    data, bbox, _ = qrDecoder.detectAndDecode(frame)

    print("DECODED DATA: ", data)

    # print(bbox)

    if not bbox is None:
        n = len(bbox)
        for j in range(n):
            cv2.line(frame, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255,0,0), 3)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    # print(cv2.getBuildInformation())




cap.release()
cv2.destroyAllWindows()
