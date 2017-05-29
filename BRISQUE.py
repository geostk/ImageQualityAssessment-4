"""
Copyright (c) 2011 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
and Center for Perceptual Systems (CPS, http://www.cps.utexas.edu) at the University of Texas at Austin (UT Austin, 
http://www.utexas.edu), is acknowledged in any publication that reports research using this code. The research
is to be cited in the bibliography as:

1) A. Mittal, A. K. Moorthy and A. C. Bovik, "BRISQUE Software Release", 
URL: http://live.ece.utexas.edu/research/quality/BRISQUE_release.zip, 2011

2) A. Mittal, A. K. Moorthy and A. C. Bovik, "No Reference Image Quality Assessment in the Spatial Domain"
submitted

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

import numpy
import cv2
from scipy import signal
from ShiftEnum import shift
from scipy.special import gamma
from scipy.misc import imresize


def gaussian_fspecial(shape=(7, 7), sigma=7 / 6):
    """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = numpy.ogrid[-m:m + 1, -n:n + 1]
    h = numpy.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < numpy.finfo(h.dtype).eps * h.max()] = 0
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
    window = window / window.sum()
    feat = []
    for i in range(0, 2):
        mu = signal.convolve2d(image, window, mode='same')
        mu_sq = numpy.multiply(mu, mu)
        sigma = numpy.sqrt(numpy.abs(signal.convolve2d(numpy.multiply(image, image), window, mode='same') - mu_sq))
        structural_distribution = numpy.divide((image - mu), (sigma + 1))
        (alpha, overall_std) = estimate_ggd_param(structural_distribution.flatten('F'))
        feat += (alpha, overall_std ** 2)
        feat = circular_shift(structural_distribution, feat)
        image = imresize(image, 0.5, interp='cubic', mode='F')
    return feat


def estimate_ggd_param(vector):
    gam = numpy.arange(0.2, 10.001, 0.001)
    r_gam = numpy.divide(numpy.multiply(gamma(numpy.divide(1, gam)), gamma(numpy.divide(3, gam))),
                         numpy.power(gamma(numpy.divide(2, gam)), 2))
    sigma_sq = numpy.mean(numpy.power(vector, 2))
    e = numpy.mean(numpy.abs(vector))
    rho = sigma_sq / (e ** 2)
    index = numpy.argmin(numpy.abs(rho - r_gam))
    return gam[index], numpy.sqrt(sigma_sq)


def estimate_asym_ggd_param(vector):
    gam = numpy.arange(0.2, 10.001, 0.001)
    r_gam = numpy.divide(numpy.power(gamma(numpy.divide(2, gam)), 2),
                         numpy.multiply(gamma(numpy.divide(1, gam)), gamma(numpy.divide(3, gam))))
    left_std = numpy.sqrt(numpy.mean(numpy.power(vector[numpy.where(vector < 0)], 2)))
    right_std = numpy.sqrt(numpy.mean(numpy.power(vector[numpy.where(vector > 0)], 2)))
    gamma_hat = left_std / right_std
    r_hat = numpy.mean(numpy.abs(vector)) ** 2 / numpy.mean(numpy.power(vector, 2))
    r_hat_norm = (r_hat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2)
    index = numpy.argmin(numpy.power(r_gam - r_hat_norm, 2))
    return gam[index], left_std, right_std


def circular_shift(structural_distribution, feat):
    shifts = [[shift.right],
              [shift.down],
              [shift.right, shift.down],
              [shift.right, shift.up]]
    shifted_structural_distribution = structural_distribution
    for i in range(0, len(shifts)):
        for j in range(0, len(shifts[i])):
            shifted_structural_distribution = numpy.roll(
                shifted_structural_distribution if j == 1 else structural_distribution, shifts[i][j].value[0],
                shifts[i][j].value[1])
        pair = numpy.multiply(structural_distribution.flatten('F'), shifted_structural_distribution.flatten('F'))
        (alpha, left_std, right_std) = estimate_asym_ggd_param(pair)
        const = numpy.sqrt(gamma(1 / alpha)) / numpy.sqrt(gamma(3 / alpha))
        mean_param = (right_std - left_std) * (gamma(2 / alpha) / gamma(1 / alpha)) * const
        feat += (alpha, mean_param, left_std ** 2, right_std ** 2)
    return feat
