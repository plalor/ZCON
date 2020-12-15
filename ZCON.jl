using LinearAlgebra
using Dates
using Printf
using SharedArrays

function processImage(im_H, im_L, b_H, b_L, R, E_in, E_dep, attenMat)
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
    
    attenMat : shape(n, 92) matrix where entry (i, j) = mu/rho(E_in[i], Z=j), where mu/rho
               is the mass attenuation coefficient of element Z = j at energy E = E_in[i].
               attenMat can be built using the 'XCOM' module (https://github.com/plalor/XCOM)
    
    
    Returns
    -------
    
    im_lambda : shape (h, l) approximation of the area density of pixel
    
    im_Z      : shape (h, l) approximation of the atomic number of each pixel
    
    """   
    Z1, Z2 = 1, 92
    zRange = Array(Z1:Z2)
    h, l = size(im_H)
    im_lambda = SharedArray{Float64}(h, l)
    im_Z = SharedArray{Float64}(h, l)
    t0 = datetime2unix(now())
    @sync @distributed for k=1:h*l
        T_H = im_H[k]
        T_L = im_L[k]
        lmbdas, minima = runNewton(T_H, T_L, b_H, b_L, R, E_in, E_dep, attenMat, zRange)
        idx = argmin(minima)
        im_lambda[k] = lmbdas[idx]
        im_Z[k] = zRange[idx]
    end
    
    t1 = datetime2unix(now())
    @printf("Completed in %.2f seconds\n", t1-t0)
    return im_lambda, im_Z
end

function runNewton(T_H, T_L, b_H, b_L, R, E_in, E_dep, attenMat, zRange)
    """Performs a Newton minimization on pixel T_H, T_L"""
    d_H = dot(E_dep, R*b_H)
    d_L = dot(E_dep, R*b_L)
    p = [T_H, T_L, b_H, b_L, R, E_in, E_dep, d_H, d_L, attenMat, zRange]
    n = length(zRange)
    lmbdas = zeros(n)
    minima = zeros(n)
    for i = 92:-1:1
        Z = zRange[i]
        if i == n
            lmbda = newton(0, Z, p..., 10)
        elseif Z == 1
            lmbda = newton(lmbdas[i+1], Z, p..., 5)
        else
            lmbda = newton(lmbdas[i+1], Z, p..., 1)
        end
        lmbdas[i] = lmbda
        minima[i] = chi2(lmbda, Z, p...)
    end
    return lmbdas, minima
end

function newton(lmbda, Z, T_H, T_L, b_H, b_L, R, E_in, E_dep, d_H, d_L, attenMat, zRange, nsteps = 10)
    """Performs 'nsteps' Newton Steps"""
    atten = attenMat[:,Z - zRange[1] + 1]
    for _ = 1:nsteps
        m0 = exp.(-atten * lmbda)
        m1 = -atten .* m0
        m2 = atten.^2 .* m0

        T_H_d0 = dot(E_dep, R * (m0 .* b_H)) / d_H
        T_L_d0 = dot(E_dep, R * (m0 .* b_L)) / d_L

        T_H_d1 = dot(E_dep, R * (m1 .* b_H)) / d_H
        T_L_d1 = dot(E_dep, R * (m1 .* b_L)) / d_L

        T_H_d2 = dot(E_dep, R * (m2 .* b_H)) / d_H
        T_L_d2 = dot(E_dep, R * (m2 .* b_L)) / d_L
    
        d1 = (1 - T_H^2 / T_H_d0^2) * T_H_d1 + (1 - T_L^2 / T_L_d0^2) * T_L_d1
        d2 = 2*T_H^2 * T_H_d1^2 / (T_H_d0^3) + (1 - T_H^2 / T_H_d0^2) * T_H_d2 +
             2*T_L^2 * T_L_d1^2 / (T_L_d0^3) + (1 - T_L^2 / T_L_d0^2) * T_L_d2
        
        lmbda = lmbda - d1 / d2
    end
    return lmbda
end

function chi2(lmbda, Z, T_H, T_L, b_H, b_L, R, E_in, E_dep, d_H, d_L, attenMat, zRange)
    """Computes the chi-squared error"""
    atten = attenMat[:,Z - zRange[1] + 1]
    m = exp.(-atten * lmbda)
    T_H_d0 = dot(E_dep, R * (m .* b_H)) / d_H
    T_L_d0 = dot(E_dep, R * (m .* b_L)) / d_L  
    return  (T_H_d0 - T_H)^2 / T_H_d0 + (T_L_d0 - T_L)^2 / T_L_d0
end

"Loaded ZCON.jl"