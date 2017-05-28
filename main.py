import subprocess
import cv2
from BRISQUE import brisque_score

capture = cv2.VideoCapture('udp://@233.12.166.100:1234')
next_frame = True
while True:
    if next_frame:
        ret, frame = capture.read()

        if not ret:
            capture.release()
            capture = cv2.VideoCapture('udp://@233.12.166.100:1234')
        else:
            cv2.imshow('frame', frame)
            # next_frame = False
            score = brisque_score(frame)

            test_data = open("test_data", "w")
            open("test_data_scaled", "w").close()
            open("output", "w").close()
            open("dump", "w").close()

            index = 1
            test_data.write(str(index) + " ")

            for feat in score:
                test_data.write(str(index) + ":" + str(feat) + " ")
                index += 1
            test_data.close()

            scale = ["wine svm-scale.exe -r allrange test_data >> test_data_scaled"]
            predict = ["wine svm-predict.exe -b 1 test_data_scaled allmodel output >> dump"]
            p1 = subprocess.Popen(scale, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            p1.wait()
            p2 = subprocess.Popen(predict, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            p2.wait()
            output = open("output", "r")
            skit = open("score", "a")
            skit.write('score: ' + output.read() + '\n')
            skit.close()
            output.close()

            capture.release()
            capture = cv2.VideoCapture('udp://@233.12.166.100:1234')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        next_frame = True

capture.release()
cv2.destroyAllWindows()
