import numpy as np
from time import time
from .utils import runNewton, loss, calcAttenMat

def createTables(b_H, b_L, R, E_g, E_dep, lmbdaRange, zRange):
    """   
    Creates tables of P, dP/dLmbda, and d2P/dLmbda2 for a range of area densitiies
    ('lmbdaRange') and atomic numbers ('zRange'). This allows subsequent calculations
    to use precomputed values from the tables instead of needing to redo the vector
    calculations.
    
    Parameters
    ----------
    
    b_H : shape (n,) representing the initial beam energy used to measure im_H
    
    b_L : shape (n,) representing the initial beam energy used to measure im_L
    
    R : Response matrix of shape (m, n), mapping energy deposited in the detector
        (E_dep) to incident photon energies (E_g). The energy bin widths dE_dep and
        dE_g need to be implicitly folded into the response matrix
    
    E_g : shape (n,) of gamma energy bin values of the initial beam energies b_H and b_L
    
    E_dep : shape (m,) of deposited energy bin values corresponding to the rows of
            the response matrix R
    
    lmbdaRange : shape(a,) array of 'lambda' values at which to calculate T's
    
    zRange : shape(b,) array of 'Z' values at which to calculate T's (usually 1:92)
    
    
    Returns
    -------
    
    P_H_d0 : shape (a, b) calculation of P_H for every value in 'lmbdaRange', 'zRange'
    
    P_L_d0 : shape (a, b) calculation of P_L for every value in 'lmbdaRange', 'zRange'
    
    P_H_d1 : shape (a, b) calculation of P_H' for every value in 'lmbdaRange', 'zRange'
    
    P_L_d1 : shape (a, b) calculation of P_L' for every value in 'lmbdaRange', 'zRange'
    
    P_H_d2 : shape (a, b) calculation of P_H'' for every value in 'lmbdaRange', 'zRange'
    
    P_L_d2 : shape (a, b) calculation of P_L'' for every value in 'lmbdaRange', 'zRange'
    
    """
    attenMat = calcAttenMat(E_g, zRange)
    a, b = lmbdaRange.size, zRange.size
    P_H_d0 = np.zeros((a, b))
    P_L_d0 = np.zeros((a, b))
    P_H_d1 = np.zeros((a, b))
    P_L_d1 = np.zeros((a, b))
    P_H_d2 = np.zeros((a, b))
    P_L_d2 = np.zeros((a, b))
    q = R.T @ E_dep
    d_H = np.dot(q, b_H)
    d_L = np.dot(q, b_L)
    print("Building lookup tables...")
    t0 = time()
    for i in range(a):
        for j in range(b):
            lmbda = lmbdaRange[i]
            Z = zRange[j]
            atten = attenMat[:,Z - zRange[0]]
            m0 = np.exp(-atten * lmbda)
            m1 = -atten * m0
            m2 = atten**2 * m0
            d_H0 = np.dot(q, m0 * b_H)
            d_L0 = np.dot(q, m0 * b_L)
            d_H1 = np.dot(q, m1 * b_H)
            d_L1 = np.dot(q, m1 * b_L)
            d_H2 = np.dot(q, m2 * b_H)
            d_L2 = np.dot(q, m2 * b_L)
            P_H_d0[i,j] = np.log(d_H / d_H0)
            P_L_d0[i,j] = np.log(d_L / d_L0)
            P_H_d1[i,j] = -d_H1 / d_H0
            P_L_d1[i,j] = -d_L1 / d_L0
            P_H_d2[i,j] = (d_H1**2 - d_H0 * d_H2) / d_H0**2
            P_L_d2[i,j] = (d_L1**2 - d_L0 * d_L2) / d_L0**2
    print("Completed in %.2f seconds" % (time() - t0))
    return P_H_d0, P_L_d0, P_H_d1, P_L_d1, P_H_d2, P_L_d2

def processImage(im_H, im_L, imVar_H, imVar_L, lmbdaRange, zRange, tables):
    """
    Approximates the area density (lambda) and atomic number (Z) for the given images
    
    Parameters
    ----------
    im_H : shape (h, l) image taken using an initial beam energy b_H, where each pixel
           entry is -log(T_H), where T_H is the tranmission value between 0 and 1
    
    im_L : shape (h, l) image taken using an initial beam energy b_L, where each pixel
           entry is -log(T_L), where T_L is the tranmission value between 0 and 1
           
    imVar_H : shape (h, l) variance matrix, where entry (i, j) is the variance of im_H[i,j]
    
    imVar_L : shape (h, l) variance matrix, where entry (i, j) is the variance of im_L[i,j]
    
    lmbdaRange : shape(a,) array of 'lambda' values
    
    zRange : shape(b,) array of 'Z' values (usually 1:92)
    
    tables : shape(6, a, b) table of computed values for P_H, P_L, P_H', P_L', P_H'',
             and P_L'' for all values in lmbdaRange and zRange
    
    
    Returns
    -------
    
    im_lambda : shape (h, l) approximation of the area density of pixel
    
    im_Z      : shape (h, l) approximation of the atomic number of each pixel
    
    """
    h, l = im_H.shape
    im_lambda = np.zeros((h, l))
    im_Z = np.zeros((h, l))
    print("Processing image...")
    t0 = time()
    for i in range(h):
        for j in range(l):
            lmbdas, minima = runNewton(im_H[i, j], im_L[i, j], imVar_H[i, j], \
                                       imVar_L[i, j], lmbdaRange, zRange, tables)
            idx = np.argmin(minima)
            im_lambda[i,j] = lmbdas[idx]
            im_Z[i,j] = zRange[idx]
    print("Completed in %.2f seconds" % (time() - t0))
    return im_lambda, im_Z

def createMesh(lmbdaArr, zArr, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables):
    """Calculates chi-squared on a (lmbda, Z) mesh for a single pixel"""
    a = lmbdaArr.size
    b = zArr.size
    minima = np.zeros((a, b))
    print("Creating mesh...")
    t0 = time()
    for i in range(a):
        for j in range(b):
            minima[i,j] = loss(lmbdaArr[i], zArr[j], P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables)
    print("Completed in %.2f seconds" % (time() - t0))
    return minima