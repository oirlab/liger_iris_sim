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
            result[i,:] = (1 - frac1 - frac2) * temp[istart, :]
        else:
            result[i,:] = np.sum(temp[istart:istop+1, :], axis=0) - frac1 * temp[istart, :] - frac2 * temp[istop, :]

    if total:
        return np.transpose(result)
    elif not total:
        return np.transpose(result) / (xbox * ybox)
    

def bin_psf(psf : np.ndarray, scale_in : tuple, scale_out : tuple):
    shape_in = psf.shape
    shape_out = (int(shape_in[0] * scale_in / scale_out), int(shape_in[1] * scale_in / scale_out))
    return frebin(psf, shape=shape_out, total=True)


def crop_psf(psf : np.ndarray, n_sigma : float = 10, xc : int | None = None, yc : int | None = None):

    # Generate grid of x and y coordinates
    y_indices, x_indices = np.indices(psf.shape)

    # Compute the weighted means (centroid) for x and y
    norm = np.sum(psf)
    
    # Compute or use provided centroid
    if xc is not None and yc is not None:
        x_mean = xc
        y_mean = yc
    else:
        x_mean = np.sum(x_indices * psf) / norm
        y_mean = np.sum(y_indices * psf) / norm


    # Compute the second moments
    x_second_moment = np.sum(((x_indices - x_mean) ** 2) * psf) / norm
    y_second_moment = np.sum(((y_indices - y_mean) ** 2) * psf) / norm

    # Compute the standard deviations
    x_std_dev = np.sqrt(x_second_moment)
    y_std_dev = np.sqrt(y_second_moment)

    # Compute the crop size
    yi = int(y_mean - n_sigma * y_std_dev)
    yf = int(y_mean + n_sigma * y_std_dev)
    xi = int(x_mean - n_sigma * x_std_dev)
    xf = int(x_mean + n_sigma * x_std_dev)
    yi = np.max(yi, 0)
    yf = np.min(yf, psf.shape[0])
    xi = np.max(xi, 0)
    xf = np.min(xf, psf.shape[1])
    psf_out = psf[yi:yf, xi:xf].copy()

    # Return
    return psf_out