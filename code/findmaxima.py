'''
aim: create a function which findes local maxima in the 2D array 
date: 26.01.2021
author: Alina
'''

from skimage.feature import peak_local_max


def find_local_maxima(Im):
    """ in the 2D array Im find local max

    Parameters
    ----------
    Im : 2D numpy array

    Returns
    -------
    yx : numpy array
        coordinates of local max
    """
    yx = peak_local_max(Im, min_distance=1,threshold_abs=250)
    
    return yx
    