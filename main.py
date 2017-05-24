import subprocess
import cv2
from BRISQUE import brisque_score

capture = cv2.VideoCapture('./resources/SampleVideo_1280x720_50mb.mp4')
scale = ["wine svm-scale.exe -r allrange test_data >> test_data_scaled"]
predict = ["wine svm-predict.exe -b 1 test_data_scaled allmodel output >> dump"]

next_frame = True
while True:
    if next_frame:
        ret, frame = capture.read()
        cv2.imshow('frame', frame)
        next_frame = False
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

        p = subprocess.Popen(scale, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        p = subprocess.Popen(predict, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        p.wait()
        output = open("output", "r")
        print(output.read())
        output.close()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        next_frame = True

capture.release()
cv2.destroyAllWindows()
