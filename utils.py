import numpy as np
from XCOM import MassAttenCoef

def runNewton(T_H, T_L, b_H, b_L, R, E_in, E_dep, attenMat, zRange):
    """Performs a Newton minimization on pixel T_H, T_L"""
    d_H = np.dot(E_dep, R @ b_H)
    d_L = np.dot(E_dep, R @ b_L)
    p = [T_H, T_L, b_H, b_L, R, E_in, E_dep, d_H, d_L, attenMat, zRange]
    n = zRange.size
    lmbdas = np.zeros(n)
    minima = np.zeros(n)
    for i in range(n)[::-1]:
        Z = zRange[i]
        if i == n-1:
            lmbda = newton(0, Z, *p, nsteps=10)
        elif Z == 1:
            lmbda = newton(lmbdas[i+1], Z, *p, nsteps=5)
        else:
            lmbda = newton(lmbdas[i+1], Z, *p, nsteps=1)
        lmbdas[i] = lmbda
        minima[i] = chi2(lmbda, Z, *p)
    return lmbdas, minima

def chi2(lmbda, Z, T_H, T_L, b_H, b_L, R, E_in, E_dep, d_H, d_L, attenMat, zRange):
    """Computes the chi-squared error"""
    atten = attenMat[:,Z - zRange[0]]
    m = np.exp(-atten * lmbda)
    T_H_d0 = np.dot(E_dep, R @ (m*b_H)) / d_H
    T_L_d0 = np.dot(E_dep, R @ (m*b_L)) / d_L  
    return  (T_H_d0 - T_H)**2 / T_H_d0 + (T_L_d0 - T_L)**2 / T_L_d0

def newton(lmbda, Z, T_H, T_L, b_H, b_L, R, E_in, E_dep, d_H, d_L, attenMat, zRange, nsteps = 10):
    """Performs 'nsteps' Newton Steps"""
    atten = attenMat[:,Z - zRange[0]]
    for _ in range(nsteps):
        m0 = np.exp(-atten * lmbda)
        m1 = -atten * m0
        m2 = atten**2 * m0

        T_H_d0 = np.dot(E_dep, R @ (m0*b_H)) / d_H
        T_L_d0 = np.dot(E_dep, R @ (m0*b_L)) / d_L

        T_H_d1 = np.dot(E_dep, R @ (m1*b_H)) / d_H
        T_L_d1 = np.dot(E_dep, R @ (m1*b_L)) / d_L

        T_H_d2 = np.dot(E_dep, R @ (m2*b_H)) / d_H
        T_L_d2 = np.dot(E_dep, R @ (m2*b_L)) / d_L
    
        d1 = (1 - T_H**2 / T_H_d0**2)*T_H_d1 + (1 - T_L**2 / T_L_d0**2)*T_L_d1
        d2 = 2*T_H**2*T_H_d1**2 / (T_H_d0**3) + (1 - T_H**2 / T_H_d0**2)*T_H_d2 + \
             2*T_L**2*T_L_d1**2 / (T_L_d0**3) + (1 - T_L**2 / T_L_d0**2)*T_L_d2
        
        lmbda = lmbda - d1 / d2
    return lmbda

def calcAttenMat(E_in, zRange):
    attenMat = np.zeros((E_in.size, zRange.size))
    for i in range(zRange.size):
        Z = zRange[i]
        atten = MassAttenCoef(E_in, Z)
        attenMat[:,i] = atten
    return attenMat