using NPZ
using ImageFiltering
using Distributions

int16Max = 2^16 - 1
kernel = centered(ones(3, 3) ./ 9)

function loadCargoImage(filename)
    """Loads dual energy cargo images from 'filename'"""
    im = npzread(filename)
    im_H_raw = float(im[:,:,1])
    im_L_raw = float(im[:,:,2])
    im_H = -log.(max.(im_H_raw, 1) / int16Max)
    im_L = -log.(max.(im_L_raw, 1) / int16Max)
    return im_H, im_L
end

function calcStatisticalVariance(im, ddof=1)
    """Computes the statistical variance of each pixel of an image by looking at neighbors"""
    im_mean = imfilter(im, kernel, "reflect")
    im_mean_sqr = imfilter(im.^2, kernel, "reflect")
    imVarStat = im_mean_sqr .- im_mean.^2
    cor = 9 / (9 - ddof)
    return cor * imVarStat
end

function calcIntrinsicVariance(im)
    """Calculates the intrinsic variance due to pixels being a 16 bit integer"""
    imVarIntr = exp.(im).^2 / int16Max^2
    return imVarIntr
end

function calcVariance(im, ddof=1)
    """Calculated the total variance of image 'im'"""
    return calcStatisticalVariance(im, ddof) .+ calcIntrinsicVariance(im)
end

function calcCovariance(im_H, im_L, ddof=1)
    """Calculates the covariance between im_H and im_L"""
    imH_mean = imfilter(im_H, kernel, "reflect")
    imL_mean = imfilter(im_L, kernel, "reflect")
    imHL_mean = imfilter(im_H .* im_L, kernel, "reflect")
    imCov = imHL_mean .- imH_mean .* imL_mean
    cor = 9 / (9 - ddof)
    return cor * imCov
end

function calcCovMat(im_H, im_L, ddof=1)
    """Calculates the covariance matrix between im_H and im_L"""
    imVar_H = calcVariance(im_H, ddof)
    imVar_L = calcVariance(im_L, ddof)
    imCov_HL = calcCovariance(im_H, im_L, ddof)
    covMat = hcat(imVar_H, imCov_HL, imCov_HL, imVar_L)
    return reshape(covMat, size(im_H)..., 2, 2)
end

function resample(im_H, im_L, n=1)
    """Resamples 'n' new images"""
    covMat = calcCovMat(im_H, im_L)
    h, l = size(im_H)
    im_H_resamp = zeros((n, h, l))
    im_L_resamp = zeros((n, h, l))
    for i = 1:h
        for j = 1:l
            mu = [im_H[i, j], im_L[i, j]]
            cov = covMat[i, j, :, :]
            resamp = max.(rand(MvNormal(mu, cov), n), 0)
            im_H_resamp[:,i,j] = resamp[1,:]
            im_L_resamp[:,i,j] = resamp[2,:]
        end
    end    
    return im_H_resamp, im_L_resamp
end