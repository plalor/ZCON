import numpy as np
from matplotlib import pyplot as plt
int16Max = 2**16 - 1

def loadCargoImage(filename):
    """Loads 6 and 4 MeV cargo images"""
    im = np.load(filename)
    im_6MeV = im[:,:,0].astype(float)
    im_4MeV = im[:,:,1].astype(float)
    return im_6MeV, im_4MeV

def preprocess(im, energy):
    """Replaces blank pixels with the known max penetration"""
    mueff = getMueff(energy)
    maxPen = getMaxPen(energy)
    im = im.copy()
    minPixel = int16Max * np.exp(-mueff * maxPen)
    im[im < minPixel] = minPixel
    return im

def convertToTransmission(im):
    """Given an image 'im' with pixel values ranging from
    0 to (2^16 - 1), converts the image into a new image
    where each entry is the transmission"""
    return im / int16Max

def convertToCMSteel(im, energy):
    """Given an image 'im' with pixel values ranging from
    0 to (2^16 - 1), converts the image into a new image
    where each entry is the 'cm-steel' equivalent"""
    mueff = getMueff(energy)   
    return -1/mueff * np.log(im / int16Max)
        
def getMaxPen(energy):
    """Returns the scanners max penetration for the given energy"""
    if energy == "6MeV":
        return 31.0
    elif energy == "4MeV":
        return 31.0
    else:
        raise ValueError("'energy' not understood")
        
def getMueff(energy):
    """Returns the effective attenuation (mueff) for the given energy"""
    if energy == "6MeV":
        return 0.3709 # Brian's value
    elif energy == "4MeV":
        return 0.4039
    else:
        raise ValueError("'energy' not understood")
    return mueff