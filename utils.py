import numpy as np
from XCOM import MassAttenCoef, MassEnergyAbsorpCoef

def runNewton(P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables):
    """Performs a Newton minimization on pixel P_H, P_L"""
    b = zRange.size
    lmbdas = np.zeros(b)
    minima = np.zeros(b)
    for i in range(b):
        Z = zRange[i]
        if i == 0:
            lmbda = newton(0, Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables, nsteps=5)
        elif i == 1:
            lmbda = newton(lmbdas[i-1], Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables, nsteps=3)
        else:
            lmbda = newton(lmbdas[i-1], Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables, nsteps=1)
        lmbdas[i] = lmbda
        minima[i] = loss(lmbda, Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables)
    return lmbdas, minima

def newton(lmbda, Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables, nsteps = 10):
    """Performs 'nsteps' Newton Steps"""
    for _ in range(nsteps):
        P_H0, P_L0, P_H1, P_L1, P_H2, P_L2 = lookup(lmbda, Z, lmbdaRange, zRange, tables)
        d1 = (2 / Var_H) * (P_H0 - P_H) * P_H1 + (2 / Var_L) * (P_L0 - P_L) * P_L1
        d2 = (2 / Var_H) * (P_H0 - P_H) * P_H2 + (2 / Var_H) * P_H1**2 + \
             (2 / Var_L) * (P_L0 - P_L) * P_L2 + (2 / Var_L) * P_L1**2
        lmbda = lmbda - d1 / d2
    return lmbda

def loss(lmbda, Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables):
    """Computes the loss"""
    P_H0, P_L0, P_H1, P_L1, P_H2, P_L2 = lookup(lmbda, Z, lmbdaRange, zRange, tables)
    return (P_H0 - P_H)**2 / Var_H + (P_L0 - P_L)**2 / Var_L

def lookup(lmbda, Z, lmbdaRange, zRange, tables):
    """Lookup P, dP/dLmbda, and d2P/dLmbda2 from tables"""
    lmbda = max(0, min(lmbda, lmbdaRange[-1])) # must be within table
    f = (lmbda - lmbdaRange[0]) / (lmbdaRange[-1] - lmbdaRange[0])
    idx1 = np.rint(f * (lmbdaRange.size - 1)).astype('int')
    idx2 = Z - zRange[0]
    P_H0 = tables[0][idx1, idx2]
    P_L0 = tables[1][idx1, idx2]
    P_H1 = tables[2][idx1, idx2]
    P_L1 = tables[3][idx1, idx2]
    P_H2 = tables[4][idx1, idx2]
    P_L2 = tables[5][idx1, idx2]
    return P_H0, P_L0, P_H1, P_L1, P_H2, P_L2

def calcAttenMat(E_g, zRange):
    attenMat = np.zeros((E_g.size, zRange.size))
    for i in range(zRange.size):
        Z = zRange[i]
        atten = MassAttenCoef(E_g, Z)
        attenMat[:,i] = atten
    return attenMat

def calcAbsorpMat(E_g, zRange):
    absorpMat = np.zeros((E_g.size, zRange.size))
    for i in range(zRange.size):
        Z = zRange[i]
        absorp = MassEnergyAbsorpCoef(E_g, Z)
        absorpMat[:,i] = absorp
    return absorpMat