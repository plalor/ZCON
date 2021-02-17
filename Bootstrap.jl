using Distributed
addprocs(2)

@everywhere begin
    include("ZCON.jl")
    include("imageprocessing.jl")
    nBootstrap = 20
    imID = 1
    loadTables = true
    zRange = Array(1:92)
    lmbdaRange = Array(LinRange(0, 300, Int(1e4)))
end

im_H, im_L = loadCargoImage("data/$(imID).npy")
imVar_H = calcVariance(im_H)
imVar_L = calcVariance(im_L)
b_H = npzread("data/b6MeV.npy")
b_L = npzread("data/b4MeV.npy")
R = npzread("data/R.npy")
E_g = npzread("data/E_g.npy")
E_dep = npzread("data/E_dep.npy")
attenMat = npzread("data/attenMat.npy")

if loadTables
    tables = npzread("data/tables.npy")
    tables = tuple([tables[idx,:,:] for idx = 1:size(tables, 1)]...)
else
    tables = createTables(b_H, b_L, R, E_g, E_dep, attenMat, lmbdaRange, zRange)
end

@printf("Resampling %d images...", nBootstrap)
im_H_resamp, im_L_resamp = resample(im_H, im_L, nBootstrap)
@printf("Completed\n")

for bootstrapID = 1:nBootstrap
    @printf("Iteration %d of %d: ", bootstrapID, nBootstrap)
    im_lambda, im_Z = processImage(im_H_resamp[bootstrapID,:,:], im_L_resamp[bootstrapID,:,:],
                                   imVar_H, imVar_L, lmbdaRange, zRange, tables)
    npzwrite("data/bootstrap/$(imID)/bootstrap$(bootstrapID)_lambda.npy", im_lambda)
    npzwrite("data/bootstrap/$(imID)/bootstrap$(bootstrapID)_Z.npy", im_Z)
end