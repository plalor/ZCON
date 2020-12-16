using LinearAlgebra
using Dates
using Printf
using SharedArrays

function createTables(b_H, b_L, R, E_in, E_dep, attenMat, lmbdaRange, zRange)
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
    a = length(lmbdaRange)
    b = length(zRange)
    T_H_d0 = zeros(a, b)
    T_L_d0 = zeros(a, b)
    T_H_d1 = zeros(a, b)
    T_L_d1 = zeros(a, b)
    T_H_d2 = zeros(a, b)
    T_L_d2 = zeros(a, b)
    d_H = dot(E_dep, R * b_H)
    d_L = dot(E_dep, R * b_L)
    @printf("Building lookup tables...")
    t0 = datetime2unix(now())
    for i in 1:a
        for j in 1:b
            lmbda = lmbdaRange[i]
            Z = zRange[j]
            atten = attenMat[:,Z - zRange[1] + 1]
            m0 = exp.(-atten * lmbda)
            m1 = -atten .* m0
            m2 = atten.^2 .* m0
            T_H_d0[i,j] = dot(E_dep, R * (m0 .* b_H)) / d_H
            T_L_d0[i,j] = dot(E_dep, R * (m0 .* b_L)) / d_L
            T_H_d1[i,j] = dot(E_dep, R * (m1 .* b_H)) / d_H
            T_L_d1[i,j] = dot(E_dep, R * (m1 .* b_L)) / d_L
            T_H_d2[i,j] = dot(E_dep, R * (m2 .* b_H)) / d_H
            T_L_d2[i,j] = dot(E_dep, R * (m2 .* b_L)) / d_L
        end
    end
    t1 = datetime2unix(now())
    @printf("Completed in %d seconds\n", t1 - t0)
    return T_H_d0, T_L_d0, T_H_d1, T_L_d1, T_H_d2, T_L_d2
end

function processImage(im_H, im_L, lmbdaRange, zRange, tables)
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
    h, l = size(im_H)
    im_lambda = SharedArray{Float64}(h, l)
    im_Z = SharedArray{Float64}(h, l)
    t0 = datetime2unix(now())
    @sync @distributed for k=1:h*l
        lmbdas, minima = runNewton(im_H[k], im_L[k], lmbdaRange, zRange, tables)
        idx = argmin(minima)
        im_lambda[k] = lmbdas[idx]
        im_Z[k] = zRange[idx]
    end
    t1 = datetime2unix(now())
    @printf("Completed in %d seconds\n", t1 - t0)
    return im_lambda, im_Z
end

function runNewton(T_H, T_L, lmbdaRange, zRange, tables)
    """Performs a Newton minimization on pixel T_H, T_L"""
    b = length(zRange)
    lmbdas = zeros(b)
    minima = zeros(b)
    for i = b:-1:1
        Z = zRange[i]
        if i == b
            lmbda = newton(0, Z, T_H, T_L, lmbdaRange, zRange, tables, 10)
        elseif Z == 1
            lmbda = newton(lmbdas[i+1], Z, T_H, T_L, lmbdaRange, zRange, tables, 5)
        else
            lmbda = newton(lmbdas[i+1], Z, T_H, T_L, lmbdaRange, zRange, tables, 1)
        end
        lmbdas[i] = lmbda
        minima[i] = chi2(lmbda, Z, T_H, T_L, lmbdaRange, zRange, tables)
    end
    return lmbdas, minima
end

function lookup(lmbda, Z, lmbdaRange, zRange, tables)
    """Calculates T, dT/dLmbda, and d2T/dLmbda2 from tables"""
    lmbda = max(0, min(lmbda, lmbdaRange[end])) # must be within table
    a = length(lmbdaRange)
    f = (lmbda - lmbdaRange[1]) / (lmbdaRange[end] - lmbdaRange[1])
    idx1 = Int(1 + round(f * (a - 1)))
    idx2 = Z - zRange[1] + 1
    T_H_d0, T_L_d0, T_H_d1, T_L_d1, T_H_d2, T_L_d2 = tables
    T_H0 = T_H_d0[idx1, idx2]
    T_L0 = T_L_d0[idx1, idx2]
    T_H1 = T_H_d1[idx1, idx2]
    T_L1 = T_L_d1[idx1, idx2]
    T_H2 = T_H_d2[idx1, idx2]
    T_L2 = T_L_d2[idx1, idx2]        
    return T_H0, T_L0, T_H1, T_L1, T_H2, T_L2
end

function newton(lmbda, Z, T_H, T_L, lmbdaRange, zRange, tables, nsteps)
    """Performs 'nsteps' Newton Steps"""
    for _ = 1:nsteps
        T_H0, T_L0, T_H1, T_L1, T_H2, T_L2 = lookup(lmbda, Z, lmbdaRange, zRange, tables)
    
        d1 = (1 - T_H^2 / T_H0^2) * T_H1 + (1 - T_L^2 / T_L0^2) * T_L1
        d2 = 2*T_H^2 * T_H1^2 / (T_H0^3) + (1 - T_H^2 / T_H0^2) * T_H2 +
             2*T_L^2 * T_L1^2 / (T_L0^3) + (1 - T_L^2 / T_L0^2) * T_L2
        
        lmbda = lmbda - d1 / d2
    end
    return lmbda
end

function chi2(lmbda, Z, T_H, T_L, lmbdaRange, zRange, tables)
    """Computes the chi-squared error"""
    T_H0, T_L0, T_H1, T_L1, T_H2, T_L2 = lookup(lmbda, Z, lmbdaRange, zRange, tables)
        
    return  (T_H0 - T_H)^2 / T_H0 + (T_L0 - T_L)^2 / T_L0
end