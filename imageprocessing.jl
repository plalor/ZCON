using NPZ
using Statistics
using Distributions

int16Max = 2^16 - 1

function loadCargoImage(filename)
    """Loads dual energy cargo images from 'filename'"""
    im = npzread(filename)
    im_H_raw = float(im[:,:,1])
    im_L_raw = float(im[:,:,2])
    im_H = -log.(im_H_raw / int16Max)
    im_L = -log.(im_L_raw / int16Max)
    return im_H, im_L
end

function calcVariance(im)
    """Calculated the total variance of image 'im'"""
    h, l = size(im)
    imVar = zeros(h, l)
    for i = 1:h
        for j = 1:l
            if !isfinite(im[i,j])
                imVar[i,j] = Inf
                continue
            else
                i0 = max(i-1, 1)
                i1 = min(i+1, h)
                j0 = max(j-1, 1)
                j1 = min(j+1, l)
                neighborhood = im[i0:i1, j0:j1]
                mask = isfinite.(neighborhood)
                var_stat = var(neighborhood[mask])
                var_intr = exp(im[i,j])^2 / int16Max^2
                imVar[i,j] = var_stat + var_intr
            end
        end
    end
    return imVar
end

function calcCovMat(im_H, im_L)
    """Calculates the covariance between im_H and im_L"""
    h, l = size(im_H)
    covMat = zeros(h, l, 2, 2)
    for i = 1:h
        for j = 1:l
            if !isfinite(im_H[i,j]) | !isfinite(im_L[i,j])
                covMat[i, j, :, :] = [Inf 0; 0 Inf]
            else
                i0 = max(i-1, 1)
                i1 = min(i+1, h)
                j0 = max(j-1, 1)
                j1 = min(j+1, l)
                neighborhood_H = vcat(im_H[i0:i1, j0:j1]...)
                neighborhood_L = vcat(im_L[i0:i1, j0:j1]...)
                mask = isfinite.(neighborhood_H) .& isfinite.(neighborhood_L)
                var_H = var(neighborhood_H[mask]) + exp(im_H[i,j])^2 / int16Max^2
                var_L = var(neighborhood_L[mask]) + exp(im_L[i,j])^2 / int16Max^2
                cov_HL = cov(neighborhood_H[mask], neighborhood_L[mask])
                covMat[i, j, :, :] = [var_H cov_HL; cov_HL var_L]
            end
        end
    end
    return covMat
end

function resample(im_H, im_L, n=1)
    """Resamples 'n' new images"""
    covMat = calcCovMat(im_H, im_L)
    h, l = size(im_H)
    im_H_resamp = zeros(n, h, l)
    im_L_resamp = zeros(n, h, l)
    for i = 1:h
        for j = 1:l
            mu = [im_H[i, j], im_L[i, j]]
            cov = covMat[i, j, :, :]
            if !all(isfinite, mu) & !all(isfinite, cov)
                im_H_resamp[:,i,j] .= im_H[i,j]
                im_L_resamp[:,i,j] .= im_L[i,j]
            else
                resamp = max.(rand(MvNormal(mu, cov), n), 0)
                im_H_resamp[:,i,j] = resamp[1,:]
                im_L_resamp[:,i,j] = resamp[2,:]
            end
        end
    end    
    return im_H_resamp, im_L_resamp
end