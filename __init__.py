import numpy as np
from time import time
from .utils import runNewton, chi2, calcAttenMat

def createTables(b_H, b_L, R, E_in, E_dep, lmbdaRange, zRange):
    """
    Creates tables of T, dT/dLmbda, and d2T/dLmbda2 for a range of area densitiies
    ('lmbdaRange') and atomic numbers ('zRange'). This allows subsequent calculations
    to use precomputed values from the tables instead of needing to redo the matrix
    calculations.
    
    Parameters
    ----------
            
    b_H : shape (n,) representing the initial beam energy used to measure im_H
    
    b_L : shape (n,) representing the initial beam energy used to measure im_L
    
    R : Response matrix of shape (m, n), mapping energy deposited in the detector 
        (E_dep) to incident photon energies (E_in). The energy bin widths dE_dep and
        dE_in need to be implicitly folded into the response matrix
                
    E_in : shape (n,) of energy bin values of the initial beam energies b_H and b_L
    
    E_dep : shape (m,) of energy bin values corresponding to the rows of the response
            matrix R
            
    lmbdaRange : shape(a,) array of 'lambda' values at which to calculate T's
    
    zRange : shape(b,) array of 'Z' values at which to calculate T's (usually 1:92)
    
    
    Returns
    -------
    
    T_H_d0 : shape (a, b) calculation of T_H for every value in 'lmbdaRange', 'zRange'
    
    T_L_d0 : shape (a, b) calculation of T_L for every value in 'lmbdaRange', 'zRange'
    
    T_H_d1 : shape (a, b) calculation of T_H' for every value in 'lmbdaRange', 'zRange'
    
    T_L_d1 : shape (a, b) calculation of T_L' for every value in 'lmbdaRange', 'zRange'
    
    T_H_d2 : shape (a, b) calculation of T_H'' for every value in 'lmbdaRange', 'zRange'
    
    T_L_d2 : shape (a, b) calculation of T_L'' for every value in 'lmbdaRange', 'zRange'
    
    """
    attenMat = calcAttenMat(E_in, zRange)
    a, b = lmbdaRange.size, zRange.size
    T_H_d0 = np.zeros((a, b))
    T_L_d0 = np.zeros((a, b))
    T_H_d1 = np.zeros((a, b))
    T_L_d1 = np.zeros((a, b))
    T_H_d2 = np.zeros((a, b))
    T_L_d2 = np.zeros((a, b))
    d_H = np.dot(E_dep, R @ b_H)
    d_L = np.dot(E_dep, R @ b_L)
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
            T_H_d0[i,j] = np.dot(E_dep, R @ (m0*b_H)) / d_H
            T_L_d0[i,j] = np.dot(E_dep, R @ (m0*b_L)) / d_L
            T_H_d1[i,j] = np.dot(E_dep, R @ (m1*b_H)) / d_H
            T_L_d1[i,j] = np.dot(E_dep, R @ (m1*b_L)) / d_L
            T_H_d2[i,j] = np.dot(E_dep, R @ (m2*b_H)) / d_H
            T_L_d2[i,j] = np.dot(E_dep, R @ (m2*b_L)) / d_L
    print("Completed in %.2f seconds" % (time() - t0))
    return T_H_d0, T_L_d0, T_H_d1, T_L_d1, T_H_d2, T_L_d2

def processImage(im_H, im_L, lmbdaRange, zRange, tables):
    """
    Approximates the area density (lambda) and atomic number (Z) for the given images
    
    Parameters
    ----------
    im_H : shape (h, l) image taken using an initial beam energy b_H, where each pixel 
           entry is a tranmission value between 0 and 1
            
    im_L : shape (h, l) image taken using an initial beam energy b_L, where each pixel 
           entry is a tranmission value between 0 and 1
            
    lmbdaRange : shape(a,) array of 'lambda' values
    
    zRange : shape(b,) array of 'Z' values (usually 1:92)
    
    tables : shape(6, a, b) table of computed values for T_H, T_L, T_H', T_L', T_H'',
             and T_L'' for all values in lmbdaRange and zRange
    
    
    Returns
    -------
    
    im_lambda : shape (h, l) approximation of the area density of pixel
    
    im_Z      : shape (h, l) approximation of the atomic number of each pixel
    
    """
    h, l = im_H.shape
    im_lambda = np.zeros((h, l))
    im_Z = np.zeros((h, l))
    t0 = time()
    for i in range(h):
        print("Processing image row %d of %d..." % (i+1, h))
        for j in range(l):
            lmbdas, minima = runNewton(im_H[i, j], im_L[i, j], lmbdaRange, zRange, tables)
            idx = np.argmin(minima)
            im_lambda[i,j] = lmbdas[idx]
            im_Z[i,j] = zRange[idx]
    print("Completed in %.2f seconds" % (time() - t0))
    return im_lambda, im_Z

def createMesh(T_H, T_L, lmbdaRange, zRange, tables):
    """Calculates chi-squared on a (lmbda, Z) mesh for a single pixel"""
    a = lmbdaRange.size
    b = zRange.size
    minima = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            minima[i,j] = chi2(lmbdaRange[i], zRange[j], T_H, T_L, lmbdaRange, zRange, tables)
    return minima