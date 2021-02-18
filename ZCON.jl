using LinearAlgebra
using Dates
using Printf
using SharedArrays

function createTables(b_H, b_L, R, E_g, E_dep, attenMat, lmbdaRange, zRange)
    """
    Creates tables of P, dP/dLmbda, and d2P/dLmbda2 for a range of area densitiies
    ('lmbdaRange') and atomic numbers ('zRange'). This allows subsequent calculations
    to use precomputed values from the tables instead of needing to redo the matrix
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
    a = length(lmbdaRange)
    b = length(zRange)
    P_H_d0 = zeros(a, b)
    P_L_d0 = zeros(a, b)
    P_H_d1 = zeros(a, b)
    P_L_d1 = zeros(a, b)
    P_H_d2 = zeros(a, b)
    P_L_d2 = zeros(a, b)
    q = transpose(R) * E_dep
    d_H = dot(q, b_H)
    d_L = dot(q, b_L)
    @printf("Building lookup tables...")
    t0 = datetime2unix(now())
    for i = 1:a
        for j = 1:b
            lmbda = lmbdaRange[i]
            Z = zRange[j]
            atten = attenMat[:,Z - zRange[1] + 1]
            m0 = exp.(-atten * lmbda)
            m1 = -atten .* m0
            m2 = atten.^2 .* m0
            d_H0 = dot(q, m0 .* b_H)
            d_L0 = dot(q, m0 .* b_L)
            d_H1 = dot(q, m1 .* b_H)
            d_L1 = dot(q, m1 .* b_L)
            d_H2 = dot(q, m2 .* b_H)
            d_L2 = dot(q, m2 .* b_L)
            P_H_d0[i,j] = log(d_H / d_H0)
            P_L_d0[i,j] = log(d_L / d_L0)
            P_H_d1[i,j] = -d_H1 / d_H0
            P_L_d1[i,j] = -d_L1 / d_L0
            P_H_d2[i,j] = (d_H1^2 - d_H0 * d_H2) / d_H0^2
            P_L_d2[i,j] = (d_L1^2 - d_L0 * d_L2) / d_L0^2
        end
    end
    t1 = datetime2unix(now())
    @printf("Completed in %d seconds\n", t1 - t0)
    return P_H_d0, P_L_d0, P_H_d1, P_L_d1, P_H_d2, P_L_d2
end

function processImage(im_H, im_L, imVar_H, imVar_L, lmbdaRange, zRange, tables)
    """
    Approximates the area density (lambda) and atomic number (Z) for the given images
    
    Parameters
    ----------
    im_H : shape (h, l) image taken using an initial beam energy b_H, where each pixel
           entry is P_H = -log(T_H), where T_H is the tranmission value between 0 and 1
    
    im_L : shape (h, l) image taken using an initial beam energy b_L, where each pixel
           entry is P_L = -log(T_L), where T_L is the tranmission value between 0 and 1
    
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
    h, l = size(im_H)
    im_lambda = SharedArray{Float64}(h, l)
    im_Z = SharedArray{Float64}(h, l)
    @printf("Processing image...")
    t0 = datetime2unix(now())
    @sync @distributed for k=1:h*l
        if !isfinite(im_H[k]) | !isfinite(im_L[k]) | !isfinite(imVar_H[k]) | !isfinite(imVar_L[k])
            im_lambda[k] = 0
            im_Z[k] = 0
        else
            lmbdas, minima = runNewton(im_H[k], im_L[k], imVar_H[k], imVar_L[k], lmbdaRange, zRange, tables)
            idx = argmin(minima)
            im_lambda[k] = lmbdas[idx]
            im_Z[k] = zRange[idx]
        end
    end
    t1 = datetime2unix(now())
    @printf("Completed in %d seconds\n", t1 - t0)
    return im_lambda, im_Z
end

function runNewton(P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables)
    """Performs a Newton minimization on pixel P_H, P_L"""
    b = length(zRange)
    lmbdas = zeros(b)
    minima = zeros(b)
    for i = 1:b
        Z = zRange[i]
        if i == 1
            lmbda = newton(0, Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables, 5)
        elseif i == 2
            lmbda = newton(lmbdas[i-1], Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables, 3)
        else
            lmbda = newton(lmbdas[i-1], Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables, 1)
        end
        lmbdas[i] = lmbda
        minima[i] = loss(lmbda, Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables)
    end
    return lmbdas, minima
end

function newton(lmbda, Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables, nsteps)
    """Performs 'nsteps' Newton Steps"""
    for _ = 1:nsteps
        P_H0, P_L0, P_H1, P_L1, P_H2, P_L2 = lookup(lmbda, Z, lmbdaRange, zRange, tables)
        d1 = (2 / Var_H) * (P_H0 - P_H) * P_H1 + (2 / Var_L) * (P_L0 - P_L) * P_L1
        d2 = (2 / Var_H) * (P_H0 - P_H) * P_H2 + (2 / Var_H) * P_H1^2 + 
             (2 / Var_L) * (P_L0 - P_L) * P_L2 + (2 / Var_L) * P_L1^2
        lmbda = lmbda - d1 / d2
    end
    return lmbda
end

function loss(lmbda, Z, P_H, P_L, Var_H, Var_L, lmbdaRange, zRange, tables)
    """Computes the loss"""
    P_H0, P_L0, P_H1, P_L1, P_H2, P_L2 = lookup(lmbda, Z, lmbdaRange, zRange, tables)
    return (P_H0 - P_H)^2 / Var_H + (P_L0 - P_L)^2 / Var_L
end

function lookup(lmbda, Z, lmbdaRange, zRange, tables)
    """Lookup P, dP/dLmbda, and d2P/dLmbda2 from tables"""
    lmbda = max(0, min(lmbda, lmbdaRange[end])) # must be within table
    f = (lmbda - lmbdaRange[1]) / (lmbdaRange[end] - lmbdaRange[1])
    idx1 = Int(1 + round(f * (length(lmbdaRange) - 1)))
    idx2 = Z - zRange[1] + 1
    P_H0 = tables[1][idx1, idx2]
    P_L0 = tables[2][idx1, idx2]
    P_H1 = tables[3][idx1, idx2]
    P_L1 = tables[4][idx1, idx2]
    P_H2 = tables[5][idx1, idx2]
    P_L2 = tables[6][idx1, idx2]
    return P_H0, P_L0, P_H1, P_L1, P_H2, P_L2
end