using Distributed
addprocs(4)

@everywhere begin
    include("ZCON.jl")
    using Plots
    using NPZ
    imID = 1
    int16Max = 2^16 - 1
    im = npzread("data/$(imID).npy")
    im_H = float(im[:,:,1]) / int16Max
    im_L = float(im[:,:,2]) / int16Max
    b_H = npzread("data/b6MeV.npy")
    b_L = npzread("data/b4MeV.npy")
    R = npzread("data/R.npy")
    E_in = npzread("data/E_in.npy")
    E_dep = npzread("data/E_dep.npy")
    attenMat = npzread("data/attenMat.npy")
end

im_lambda, im_Z = processImage(im_H, im_L, b_H, b_L, R, E_in, E_dep, attenMat)
npzwrite("data/out/im$(imID)_lambda.npy", im_lambda)
npzwrite("data/out/im$(imID)_Z.npy", im_Z)

pyplot()
h1 = heatmap(reverse(im_lambda, dims=1), title = "Reconstructed Area Density")
display(h1)

h2 = heatmap(reverse(im_Z, dims=1), title = "Reconstructed Z", reuse = false)
display(h2)