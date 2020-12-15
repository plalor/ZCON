# ZCON

ZCON is a method for approximating the atomic number of materials from a radiograph. For a given pair of images taking using two different energies, ZCON calculates the area density 'lambda' and atomic number 'Z' that best reproduces the transmission values for each pixel on the input images.
    
## To run:

An example script using 4 processors is shown in JuliaExample.jl. This should be used as a template, where the npzread calls are replaced your image, beam spectrum, response matrix, and attenuation matrix. The required inputs are as follows:

* `im_H` and `im_L`, the two images taken at different energies
* `b_H` and `b_L`, the beam energies used to produce im_H and im_L
* `R`, the detector response matrix
* `E_in` and `E_dep`, the energy bin values of R and b
* `attenMat`, a matrix of mass attenuation coefficients

See the documentation in ZCON.jl for more details. The core of JuliaExample.jl are the following commands:
```julia
include("ZCON.jl")
im_lambda, im_Z = processImage(im_H, im_L, b_H, b_L, R, E_in, E_dep, attenMat)
```

A working version exists for python, although it is significantly slower. It can be run as follows:
```python
from ZCON import processImage
im_lambda, im_Z = processImage(im_H, im_L, b_H, b_L, R, E_in, E_dep)
```

## Example:

The directory 'data' contains working data using radiograph images taken from an Rapiscan Eagle R60 scanner. b_H and b_L are simulated 6 MeV and 4 MeV Bremsstrahlung beam spectra using a tungsten target backed by copper, and the response matrix R is generated from simulations of Cadmium Tungstate (CdWO4) crystals.

To run:

```console
julia -i JuliaExample.jl
```

## A note about attenMat

The input attenMat can be generated using the XCOM module (https://github.com/plalor/XCOM). XCOM currently only supports Python, so attenMat would need to be constructed and saved in Python:

```python
from XCOM import MassAttenCoef
zRange = np.arange(1, 93) # Hydrogen through Uranium
attenMat = calcAttenMat(E_in, zRange)
np.save("attenMat", attenMat)
```
and then loaded into Julia:
```Julia
using NPZ
attenMat = npzread("attenMat.npy")
```