import numpy as np
from numpy.random import multivariate_normal
from scipy.ndimage import uniform_filter

int16Max = 2**16 - 1

def loadCargoImage(filename):
    """Loads dual energy cargo images from 'filename'"""
    im = np.load(filename)
    im_H_raw = im[:,:,0].astype(float)
    im_L_raw = im[:,:,1].astype(float)
    im_H = -np.log(np.maximum(im_H_raw, 1) / int16Max)
    im_L = -np.log(np.maximum(im_L_raw, 1) / int16Max)
    return im_H, im_L

def calcStatisticalVariance(im, ddof=1):
    """Computes the statistical variance of each pixel of an image by looking at neighbors"""
    im_mean = uniform_filter(im, size = 3, mode = "mirror")
    im_mean_sqr = uniform_filter(im**2, size = 3, mode = "mirror")
    imVarStat = im_mean_sqr - im_mean**2
    cor = 9 / (9 - ddof)
    return cor * imVarStat

def calcIntrinsicVariance(im):
    """Calculates the intrinsic variance due to pixels being a 16 bit integer"""
    imVarIntr = np.exp(im)**2 / int16Max**2
    return imVarIntr

def calcVariance(im, ddof=1):
    """Calculated the total variance of image 'im'"""
    return calcStatisticalVariance(im, ddof=ddof) + calcIntrinsicVariance(im)

def calcCovariance(im_H, im_L, ddof=1):
    """Calculates the covariance between im_H and im_L"""
    imH_mean = uniform_filter(im_H, size = 3, mode = "mirror")
    imL_mean = uniform_filter(im_L, size = 3, mode = "mirror")
    imHL_mean = uniform_filter(im_H*im_L, size = 3, mode = "mirror")
    imCov = imHL_mean - imH_mean*imL_mean
    cor = 9 / (9 - ddof)
    return cor * imCov

def calcCovMat(im_H, im_L):
    """Calculates the covariance matrix between im_H and im_L"""
    imVar_H = calcVariance(im_H)
    imVar_L = calcVariance(im_L)
    imCov_HL = calcCovariance(im_H, im_L)
    covMat = np.array([[imVar_H, imCov_HL], [imCov_HL, imVar_L]])
    return np.moveaxis(covMat, [0, 1], [2, 3])

def resample(im_H, im_L, n=1):
    """Resamples 'n' new images"""
    covMat = calcCovMat(im_H, im_L)
    h, l = im_H.shape
    im_H_resamp = np.zeros((n, h, l))
    im_L_resamp = np.zeros((n, h, l))
    for i in range(h):
        for j in range(l):
            mu = (im_H[i, j], im_L[i, j])
            cov = covMat[i, j]
            resamp = np.maximum(multivariate_normal(mu, cov, size=n), 0)
            im_H_resamp[:,i,j] = resamp[:,0]
            im_L_resamp[:,i,j] = resamp[:,1]
    return im_H_resamp, im_L_resamp