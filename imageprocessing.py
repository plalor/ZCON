import numpy as np
from numpy.random import multivariate_normal
from scipy.ndimage import generic_filter

int16Max = 2**16 - 1

def loadCargoImage(filename):
    """Loads dual energy cargo images from 'filename'"""
    int16Max = 2**16 - 1
    im = np.load(filename)
    im_H_raw = im[:,:,0].astype(float)
    im_L_raw = im[:,:,1].astype(float)
    im_H = -np.log(im_H_raw / int16Max)
    im_L = -np.log(im_L_raw / int16Max)
    return im_H, im_L

def calcVariance(im, ddof=1):
    """Calculated the total variance of image 'im'"""
    h, l = im.shape
    imVar = np.zeros((h, l))
    for i in range(h):
        for j in range(l):
            if not np.isfinite(im[i,j]):
                imVar[i,j] = np.inf
                continue
            else:
                i0 = max(i-1, 0)
                i1 = min(i+2, h)
                j0 = max(j-1, 0)
                j1 = min(j+2, l)
                neighborhood = im[i0:i1, j0:j1]
                mask = np.isfinite(neighborhood)
                var_stat = np.var(neighborhood[mask], ddof=ddof)
                var_intr = np.exp(im[i,j])**2 / int16Max**2
                imVar[i,j] = var_stat + var_intr
    return imVar

def calcCovMat(im_H, im_L, ddof=1):
    h, l = im_H.shape
    covMat = np.zeros((h, l, 2, 2))
    for i in range(h):
        for j in range(l):
            if not np.isfinite(im_H[i,j]) and not np.isfinite(im_L[i, j]):
                covMat[i,j] = np.diag([np.inf, np.inf])
            else:
                i0 = max(i-1, 0)
                i1 = min(i+2, h)
                j0 = max(j-1, 0)
                j1 = min(j+2, l)
                neighborhood_H = np.ravel(im_H[i0:i1, j0:j1])
                neighborhood_L = np.ravel(im_L[i0:i1, j0:j1])
                mask = np.logical_and(np.isfinite(neighborhood_H), np.isfinite(neighborhood_L))
                covMat[i,j] = np.cov(neighborhood_H[mask], neighborhood_L[mask], ddof = ddof)
                covMat[i,j,0,0] += np.exp(im_H[i,j])**2 / int16Max**2
                covMat[i,j,1,1] += np.exp(im_L[i,j])**2 / int16Max**2
    return covMat

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
            if not np.isfinite(mu).all() or not np.isfinite(cov).all():
                im_H_resamp[:,i,j] = im_H[i, j]
                im_L_resamp[:,i,j] = im_L[i, j]
            else:
                resamp = np.maximum(multivariate_normal(mu, cov, size=n), 0)
                im_H_resamp[:,i,j] = resamp[:,0]
                im_L_resamp[:,i,j] = resamp[:,1]
    return im_H_resamp, im_L_resamp