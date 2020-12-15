import numpy as np
from time import time
from .utils import runNewton, chi2, calcAttenMat

def processImage(im_H, im_L, b_H, b_L, R, E_in, E_dep):
    """
    Approximates the area density (lambda) and atomic number (Z) for the given images
    
    Parameters
    ----------
    im_H  : shape (h, l) image taken using an initial beam energy b_H, where each pixel 
            entry is a tranmission value between 0 and 1
            
    im_L  : shape (h, l) image taken using an initial beam energy b_L, where each pixel 
            entry is a tranmission value between 0 and 1
            
    b_H   : shape (n,) representing the initial beam energy used to measure im_H
    
    b_L   : shape (n,) representing the initial beam energy used to measure im_L
    
    R     : Response matrix of shape (m, n), mapping energy deposited in the detector 
            (E_dep) to incident photon energies (E_in). The energy bin widths dE_dep and
            dE_in need to be implicitly folded into the response matrix
                
    E_in  : shape (n,) of energy bin values of the initial beam energies b_H and b_L
    
    E_dep : shape (m,) of energy bin values corresponding to the rows of the response
            matrix R
    
    
    Returns
    -------
    
    im_lambda : shape (h, l) approximation of the area density of pixel
    
    im_Z      : shape (h, l) approximation of the atomic number of each pixel
    
    """
    Z1, Z2 = 1, 92
    zRange = np.arange(Z1, Z2+1)
    attenMat = calcAttenMat(E_in, zRange)
    h, l = im_H.shape
    im_lambda = np.zeros((h, l))
    im_Z = np.zeros((h, l))
    t0 = time()
    for i in range(h):
        print("Processing image row %d of %d..." % (i+1, h))
        for j in range(l):
            T_H = im_H[i, j]
            T_L = im_L[i, j]
            lmbdas, minima = runNewton(T_H, T_L, b_H, b_L, R, E_in, E_dep, attenMat, zRange)
            idx = np.argmin(minima)
            im_lambda[i,j] = lmbdas[idx]
            im_Z[i,j] = zRange[idx]
    print("Completed in %.2f seconds" % (time() - t0))
    return im_lambda, im_Z

def createMesh(T_H, T_L, b_H, b_L, R, E_in, E_dep, lmbdaMax=250):
    """Calculates chi-squared on a a (lmbda, Z) mesh for a single pixel"""
    Z1, Z2 = 1, 92
    zRange = np.arange(Z1, Z2+1)
    attenMat = calcAttenMat(E_in, zRange)
    d_H = np.dot(E_dep, R @ b_H)
    d_L = np.dot(E_dep, R @ b_L)
    m = lmbdaMax+1
    n = zRange.size
    lmbdaRange = np.linspace(0, lmbdaMax, m)
    minima = np.zeros((m, n))
    p = [T_H, T_L, b_H, b_L, R, E_in, E_dep, d_H, d_L, attenMat, zRange]
    for i in range(m):
        for j in range(n):
            minima[i,j] = chi2(lmbdaRange[i], zRange[j], *p)
    return lmbdaRange, zRange, minima