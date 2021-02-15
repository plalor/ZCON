using Distributed
addprocs(1)

@everywhere begin
    include("ZCON.jl")
    using Plots
    using NPZ
    imID = 1
    loadTables = false
    im_H = npzread("data/im$(imID)_H.npy")
    im_L = npzread("data/im$(imID)_L.npy")
    imVar_H = npzread("data/imVar$(imID)_H.npy")
    imVar_L = npzread("data/imVar$(imID)_L.npy")
    b_H = npzread("data/b6MeV.npy")
    b_L = npzread("data/b4MeV.npy")
    R = npzread("data/R.npy")
    E_g = npzread("data/E_g.npy")
    E_dep = npzread("data/E_dep.npy")
    attenMat = npzread("data/attenMat.npy")
    if loadTables
        tables = npzread("data/tables.npy")
        tables = tuple([tables[idx,:,:] for idx in 1:size(tables, 1)]...)
    end
    zRange = Array(1:92)
    lmbdaRange = Array(LinRange(0, 300, Int(1e4)))
end

if ~loadTables
    tables = createTables(b_H, b_L, R, E_g, E_dep, attenMat, lmbdaRange, zRange)
end
im_lambda, im_Z = processImage(im_H, im_L, imVar_H, imVar_L, lmbdaRange, zRange, tables)
npzwrite("data/out/im$(imID)_lambda.npy", im_lambda)
npzwrite("data/out/im$(imID)_Z.npy", im_Z)

pyplot()
h1 = heatmap(reverse(im_lambda, dims=1), title = "Reconstructed Area Density")
display(h1)

h2 = heatmap(reverse(im_Z, dims=1), title = "Reconstructed Z", reuse = false)
display(h2)