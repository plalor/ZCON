ZCON is a method for approximating the atomic number of materials from a radiograph. For a given pair of images taking using two different energies, ZCON calculates the area density 'lambda' and atomic number 'Z' that best reproduces the transmission values for each pixel on the input images.

Requirements:
--  XCOM (https://github.com/plalor/XCOM), a module for loading in and interpolating NIST photon cross section data

Required inputs:
--  im_H and im_L, representing two images taken at different energies
--  b_H and b_L, representing the beam energies used to produce im_H and im_L
--  R, a detector response matrix
--  E_in and E_dep, representing the energy bin values of R and b
    
Usage:

>>> from ZCON import processImage
>>> im_lambda, im_Z = processImage(im_H, im_L, b_H, b_L, R, E_in, E_dep)

Examples:

CargoExample.ipynb contains a working example using radiograph images taken from an Rapiscan Eagle R60 scanner.
