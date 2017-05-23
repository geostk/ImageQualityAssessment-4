import numpy
import cv2
from scipy import signal
from ShiftEnum import shifts as shift
from scipy.special import gamma
from scipy.misc import imresize
from sklearn import svm
from sklearn import preprocessing
from training import training_data
from training import training_data2

img = cv2.imread('./resources/Lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_fspecial(shape=(7, 7), sigma=7/6):
    """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = numpy.ogrid[-m:m+1, -n:n+1]
    h = numpy.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < numpy.finfo(h.dtype).eps*h.max()] = 0
    h_sum = h.sum()
    if h_sum != 0:
        h /= h_sum
    return h


def brisque_score(image):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image.astype(float)
    return brisque_feature(image)


def brisque_feature(image):

    window = gaussian_fspecial((7, 7), 7 / 6)
    window = window/window.sum()
    feat = []
    for i in range(0, 2):
        mu = signal.convolve2d(image, window, mode='same')
        mu_sq = numpy.multiply(mu, mu)
        sigma = numpy.sqrt(
            numpy.abs(signal.convolve2d(numpy.multiply(image, image), window, mode='same') - mu_sq))
        structdis = numpy.divide((image - mu), (sigma + 1))
        (alpha, overall_std) = estimate_ggd_param(structdis.flatten('F'))
        feat += (alpha, overall_std**2)
        feat = circular_shift(structdis, feat)
        image = imresize(image, 0.5, interp='cubic', mode='F')
    return feat


def estimate_ggd_param(vector):
    gam = numpy.arange(0.2, 10.001, 0.001)
    r_gam = numpy.divide(numpy.multiply(gamma(numpy.divide(1, gam)), gamma(numpy.divide(3, gam))),
                         numpy.power(gamma(numpy.divide(2, gam)), 2))
    sigma_sq = numpy.mean(numpy.power(vector, 2))
    e = numpy.mean(numpy.abs(vector))
    rho = sigma_sq/(e**2)
    index = numpy.argmin(numpy.abs(rho - r_gam))
    return gam[index], numpy.sqrt(sigma_sq)


def estimate_aggd_param(vector):
    gam = numpy.arange(0.2, 10.001, 0.001)
    r_gam = numpy.divide(numpy.power(gamma(numpy.divide(2, gam)), 2),
                         numpy.multiply(gamma(numpy.divide(1, gam)), gamma(numpy.divide(3, gam))))
    left_std = numpy.sqrt(numpy.mean(numpy.power(vector[numpy.where(vector < 0)], 2)))
    right_std = numpy.sqrt(numpy.mean(numpy.power(vector[numpy.where(vector > 0)], 2)))
    gammahat = left_std / right_std
    rhat = numpy.mean(numpy.abs(vector)) ** 2 / numpy.mean(numpy.power(vector, 2))
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)
    index = numpy.argmin(numpy.power(r_gam - rhatnorm, 2))
    return gam[index], left_std, right_std


def circular_shift(structdis, feat):
    shifts = [[shift.right],
              [shift.down],
              [shift.right, shift.down],
              [shift.right, shift.up]]
    shifted_structdis = structdis
    for i in range(0, len(shifts)):
        for j in range(0, len(shifts[i])):
            shifted_structdis = numpy.roll(shifted_structdis if j == 1 else structdis, shifts[i][j].value[0],
                                           shifts[i][j].value[1])
        pair = numpy.multiply(structdis.flatten('F'), shifted_structdis.flatten('F'))
        (alpha, leftstd, rightstd) = estimate_aggd_param(pair)
        const = numpy.sqrt(gamma(1/alpha))/numpy.sqrt(gamma(3/alpha))
        meanparam = (rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const
        feat += (alpha, meanparam, leftstd**2, rightstd**2)
    return feat

score = numpy.array(brisque_score(gray)).reshape(-1, 1)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
skit = scaler.fit_transform(training_data, score)
print(skit)

#clf = svm.SVC()

#cmd = ["wine svm-scale -r allrange"]
#p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
# output = p.communicate()[0]
# print(output)
#
# for a in iter(p.stderr.readline, b''):
#     a = a.decode('UTF-8')
#     print(a)
