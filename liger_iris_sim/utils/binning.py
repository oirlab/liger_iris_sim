import numpy as np
import scipy.ndimage

def rebin(image, new_shape):
    M, N = image.shape
    mm, nn = new_shape
    if mm < M:
        return image.reshape((mm, int(M / mm), nn, int(N / nn))).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(image, int(mm / M), axis=0), nn / N, axis=1)


def frebin(array, new_shape, total=True):
    y, x = array.shape
    y1 = y - 1
    x1 = x - 1
    xbox = x / new_shape[0]
    ybox = y / new_shape[1]

    #Determine if integral contraction so we can use rebin
    if (x == int(x)) and (y == int(y)):
        if (x % new_shape[0] == 0) and (y % new_shape[1] == 0):
            return rebin(array, (new_shape[1], new_shape[0])) * xbox * ybox

    # Otherwise if not integral contraction
    # First bin in y dimension
    temp = np.zeros((new_shape[1], x), dtype=float)
    # Loop on output image lines
    for i in range(0, int(new_shape[1]), 1):
        rstart = i * ybox
        istart = int(rstart)
        rstop = rstart + ybox
        istop = int(rstop)
        if istop > y1:
            istop = y1
        frac1 = rstart - istart
        frac2 = 1 - (rstop - istop)
        
        #Add pixel values from istart to istop an subtract
        #fracion pixel from istart to rstart and fraction
        #fraction pixel from rstop to istop.
        if istart == istop:
            temp[i, :] = (1.0 - frac1 - frac2) * array[istart,:]
        else:
            temp[i, :] = np.sum(array[istart:istop+1, :], axis=0) - frac1 * array[istart, :] - frac2 * array[istop, :]
            
    temp = np.transpose(temp)

    #Bin in x dimension
    result = np.zeros((new_shape[0], new_shape[1]), dtype=float)

    #Loop on output image samples
    for i in range(0, int(new_shape[0]), 1):
        rstart = i*xbox
        istart = int(rstart)
        rstop = rstart + xbox
        istop = int(rstop)
        if istop > x1:
            istop = x1
        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)
        if istart == istop:
            result[i, :] = (1 - frac1 - frac2) * temp[istart, :]
        else:
            result[i, :] = np.sum(temp[istart:istop+1, :], axis=0) - frac1 * temp[istart, :] - frac2 * temp[istop, :]

    if total:
        return np.transpose(result)
    elif not total:
        return np.transpose(result) / (xbox * ybox)
    

def bin_image(image : np.ndarray, scale_in : tuple, scale_out : tuple):
    shape_in = image.shape
    shape_out = (int(shape_in[0] * scale_in / scale_out), int(shape_in[1] * scale_in / scale_out))
    return frebin(image, new_shape=shape_out, total=True)


def crop_AO_psf(
        psf : np.ndarray,
        scale : float, wavelength : float, colldiam : float, n : int | None = None,
        center : tuple[float, float] | None = None
    ):
    """
    Parameters:
    psf (np.ndarray): The PSF to resample.
    scale (float): The size of a PSF pixel in arcsec.
    colldiam (float): The effective collimating diameter
    n (float): The number of lambda / D's to crop by.
    center (float): The center of the PSF (y, x)

    Returns:
    np.ndarray: The new PSF.
    tuple[float, float]: The center of the new psf
    """
    
    # "Center" of PSF
    if center is None:
        center = ((psf.shape[0] - 1) / 2, (psf.shape[1] - 1) / 2)

    # lambda / D per pixel
    s = 206265 * wavelength / (colldiam * 1E6) / scale

    # Default number of lambda / D's?
    if n is None:
        n = 100

    # Compute the crop size
    # Initial bounds
    w = int(n * s / np.sqrt(2))
    yi = int(center[0] - w)
    yf = int(center[0] + w)
    xi = int(center[1] - w)
    xf = int(center[1] + w)

    # Check bounds
    yi = np.maximum(yi, 0)
    yf = np.minimum(yf, psf.shape[0] - 1)
    xi = np.maximum(xi, 0)
    xf = np.minimum(xf, psf.shape[1] - 1)

    # Slice PSF
    psf_out = psf[yi:yf+1, xi:xf+1].copy()

    # Return
    return center, psf_out