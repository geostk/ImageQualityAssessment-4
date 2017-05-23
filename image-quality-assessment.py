import numpy
import cv2
from skimage.measure import compare_ssim as ssim
import time

reference_img = cv2.imread('./resources/kungen.jpg')
distorted_img = cv2.imread('./resources/kungen2.png')


def mse(img1, img2):
    err = numpy.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1]) * img1.shape[2]
    return err


def psnr(img1, img2):
    return 20 * numpy.log10([255]) - 10 * numpy.log10([mse(img1, img2)])


millis = int(round(time.time() * 1000))
print(psnr(reference_img, distorted_img))
print(int(round(time.time() * 1000)) - millis)

millis = int(round(time.time() * 1000))
print(mse(reference_img, distorted_img))
print(int(round(time.time() * 1000)) - millis)



millis = int(round(time.time() * 1000))
print(ssim(reference_img, distorted_img, multichannel=True))
print(int(round(time.time() * 1000)) - millis)