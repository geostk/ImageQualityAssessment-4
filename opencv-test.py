import numpy as np
import cv2
import time

cap = cv2.VideoCapture('./resources/SampleVideo_1280x720_50mb.mp4')

#while True:
ret, frame = cap.read()
cv2.imshow('frame', frame)

rows,cols,channels = frame.shape
mat = "[ "
for i in range(rows):

    for j in range(cols):
        mat += str(frame[i, j])
        mat += ","
    mat += " ]\n"

print(mat)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
