# ZCON

ZCON is a high performance Julia module for approximating the area density and atomic number of materials from a radiograph. For a given pair of images taking using two different incident energy spectra, ZCON calculates the area density 'lambda' and atomic number 'Z' that best reproduces the transmission values for each pixel on the input images. ZCON is implemented in both Julia and Python, although it is recommended to be run in Julia due to significantly improved performance.
    
## To run:

An example script using 4 processors is shown in JuliaExample.jl. This should be used as a run template, where the npzread calls are replaced your image(s), beam spectra, response matrix, and attenuation matrix. The required inputs are as follows:

* `im_H` and `im_L`, the two images taken at different energies
* `imVar_H` and `imVar_L`, the pixel-by-pixel variance of the previous images
* `b_H` and `b_L`, the beam energies used to produce im_H and im_L
* `R`, the detector response matrix
* `E_g` and `E_dep`, the energy bin values of R and b (short for E_gamma and E_deposited)
* `attenMat`, a matrix of mass attenuation coefficients (see note below)

See the documentation in ZCON.jl for more details on the method inputs and outputs. The core of JuliaExample.jl are the following commands:
```julia
include("ZCON.jl")
tables = createTables(b_H, b_L, R, E_in, E_dep, attenMat, lmbdaRange, zRange)
im_lambda, im_Z = processImage(im_H, im_L, imVar_H, imVar_L, lmbdaRange, zRange, tables)
```

The call to `createTables` (on a mesh defined by `lmbdaRange` and `zRange`) will create a lookup table of the reconstructed pixel transmission `T_hat` and its derivatives. A typical range for `lmbdaRange` is 0 to 300 g/cm^2 and `zRange` is 1 to 92. The subsequent call to `processImage` iterates over all pixels in the image and finds the area density `lambda` and atomic number `Z` that minimize the loss.

A working version exists for python, although it is significantly slower. It can be run as follows:
```python
from ZCON import createTables, processImage
tables = createTables(b_H, b_L, R, E_in, E_dep, lmbdaRange, zRange)
im_lambda, im_Z = processImage(im_H, im_L, imVar_H, imVar_L, lmbdaRange, zRange, tables)
```

## Example:

The subdirectory 'data' contains three example images from radiographs taken by a Rapiscan Eagle R60 scanner ('1.npy', '2.npy', and '3.npy'). Furthermore, 'data' contains 'b_H.npy' and 'b_L.npy', which are simulated 6 MeV and 4 MeV Bremsstrahlung beam spectra using a tungsten target backed by copper. The response matrix 'R.npy' is generated from simulations of Cadmium Tungstate (CdWO4) crystals with binning defined in 'E_g.npy' and 'E_dep.npy'.

To run:

```console
julia -i JuliaExample.jl
```

The subdirectory 'data/out' contains the outputs of the runs as '.npy' files along with visualizations in '.png' format if you just want to view the results.

## A note about attenMat

The input `attenMat` can be generated using the XCOM module (https://github.com/plalor/XCOM). A function `calcAttenMat` to do so is included in 'utils.py'. XCOM currently only supports Python, so `attenMat` would need to be constructed and saved in Python:
```python
from utils import calcAttenMat
zRange = np.arange(1, 93) # Hydrogen through Uranium
attenMat = calcAttenMat(E_in, zRange)
np.save("attenMat", attenMat)
```
and then loaded into Julia:
```Julia
using NPZ
attenMat = npzread("attenMat.npy")
```
