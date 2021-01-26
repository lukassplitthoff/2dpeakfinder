'''
aim: create a function which would load the .mat file 
date: 25.01.2021
author: Alina
'''
import scipy.io as sio

def load_image(path):
    """Load the image and display it 

    Parameters
    ----------
    path : string
        path to the file

    Returns
    -------
    Im : numpy array
    """
    Im = sio.loadmat(path)
    return Im['Im']
    



