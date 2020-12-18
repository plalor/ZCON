import numpy as np
from XCOM import MassAttenCoef, MassEnergyAbsorpCoef

def runNewton(T_H, T_L, lmbdaRange, zRange, tables):
    """Performs a Newton minimization on pixel T_H, T_L"""
    b = zRange.size
    lmbdas = np.zeros(b)
    minima = np.zeros(b)
    for i in range(b)[::-1]:
        Z = zRange[i]
        if i == b-1:
            lmbda = newton(0, Z, T_H, T_L, lmbdaRange, zRange, tables, nsteps=10)
        elif Z == 1:
            lmbda = newton(lmbdas[i+1], Z, T_H, T_L, lmbdaRange, zRange, tables, nsteps=5)
        else:
            lmbda = newton(lmbdas[i+1], Z, T_H, T_L, lmbdaRange, zRange, tables, nsteps=1)
        lmbdas[i] = lmbda
        minima[i] = chi2(lmbda, Z, T_H, T_L, lmbdaRange, zRange, tables)
    return lmbdas, minima

def newton(lmbda, Z, T_H, T_L, lmbdaRange, zRange, tables, nsteps = 10):
    """Performs 'nsteps' Newton Steps"""
    for _ in range(nsteps):
        T_H0, T_L0, T_H1, T_L1, T_H2, T_L2 = lookup(lmbda, Z, lmbdaRange, zRange, tables)
        d1 = (1 - T_H**2 / T_H0**2)*T_H1 + (1 - T_L**2 / T_L0**2)*T_L1
        d2 = 2*T_H**2 * T_H1**2 / (T_H0**3) + (1 - T_H**2 / T_H0**2)*T_H2 + \
             2*T_L**2 * T_L1**2 / (T_L0**3) + (1 - T_L**2 / T_L0**2)*T_L2
        lmbda = lmbda - d1 / d2
    return lmbda

def lookup(lmbda, Z, lmbdaRange, zRange, tables):
    """Calculates T, dT/dLmbda, and d2T/dLmbda2 from tables"""
    lmbda = max(0, min(lmbda, lmbdaRange[-1])) # must be within table
    f = (lmbda - lmbdaRange[0]) / (lmbdaRange[-1] - lmbdaRange[0])
    idx1 = np.rint(f * (lmbdaRange.size - 1)).astype('int')
    idx2 = Z - zRange[0]
    T_H0 = tables[0][idx1, idx2]
    T_L0 = tables[1][idx1, idx2]
    T_H1 = tables[2][idx1, idx2]
    T_L1 = tables[3][idx1, idx2]
    T_H2 = tables[4][idx1, idx2]
    T_L2 = tables[5][idx1, idx2]
    return T_H0, T_L0, T_H1, T_L1, T_H2, T_L2

def chi2(lmbda, Z, T_H, T_L, lmbdaRange, zRange, tables):
    """Computes the chi-squared error"""
    T_H0, T_L0, T_H1, T_L1, T_H2, T_L2 = lookup(lmbda, Z, lmbdaRange, zRange, tables) 
    return  (T_H0 - T_H)**2 / T_H0 + (T_L0 - T_L)**2 / T_L0

def loadCargoImage(filename):
    """Loads 6 and 4 MeV cargo images from '1.npy', '2.npy', and '3.npy'"""
    int16Max = 2**16 - 1
    im = np.load(filename)
    im_6MeV = im[:,:,0].astype(float)
    im_4MeV = im[:,:,1].astype(float)
    return im_6MeV / int16Max, im_4MeV / int16Max

def calcAttenMat(E_in, zRange):
    attenMat = np.zeros((E_in.size, zRange.size))
    for i in range(zRange.size):
        Z = zRange[i]
        atten = MassAttenCoef(E_in, Z)
        attenMat[:,i] = atten
    return attenMat

def calcAbsorpMat(E_in, zRange):
    absorpMat = np.zeros((E_in.size, zRange.size))
    for i in range(zRange.size):
        Z = zRange[i]
        absorp = MassEnergyAbsorpCoef(E_in, Z)
        absorpMat[:,i] = absorp
    return absorpMat