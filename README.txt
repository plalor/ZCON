ZCON is a method for approximating the atomic number of materials from a radiograph. For a given pair of images taking using two different energies, ZCON calculates the area density 'lambda' and atomic number 'Z' that best reproduces the transmission values for each pixel on the input images.

Requirements:
--  XCOM (https://github.com/plalor/XCOM), a module for loading in and interpolating NIST photon cross section data in order to calculate attenMat

Required inputs:
--  im_H and im_L, representing two images taken at different energies
--  b_H and b_L, representing the beam energies used to produce im_H and im_L
--  R, a detector response matrix
--  E_in and E_dep, representing the energy bin values of R and b
    
Usage:

In Python:

>>> from ZCON import processImage
>>> im_lambda, im_Z = processImage(im_H, im_L, b_H, b_L, R, E_in, E_dep)

In Julia:

> include("ZCON.jl")
> im_lambda, im_Z = processImage(im_H, im_L, b_H, b_L, R, E_in, E_dep, attenMat)

Note: the Julia version runs approximately 15x faster than the Python version.

Examples:

The directory 'data' contains a working example using radiograph images taken from an Rapiscan Eagle R60 scanner. b_H and b_L are simulated 6 MeV and 4 MeV Bremsstrahlung beam spectra using a tungsten target backed by copper, and the response matrix R is generated from simulations of Cadmium Tungstate (CdWO4) crystals.

To run:

julia -i JuliaExample.jl

MPI: mpiexecjl -n 4 julia MPI_helloworld.jl