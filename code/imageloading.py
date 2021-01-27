'''
<<<<<<< HEAD
aim: create a function which would load the .mat file
=======
aim: create a function which would load the .mat file 
>>>>>>> origin/peakgenerator
date: 25.01.2021
author: Alina
'''
import scipy.io as sio

def load_image(path):
<<<<<<< HEAD
    """Load the image and display it
=======
    """Load the image and display it 

>>>>>>> origin/peakgenerator
    Parameters
    ----------
    path : string
        path to the file
<<<<<<< HEAD
=======

>>>>>>> origin/peakgenerator
    Returns
    -------
    Im : numpy array
    """
    Im = sio.loadmat(path)
<<<<<<< HEAD
    return Im['Im']
=======
    return Im['Im']
>>>>>>> origin/peakgenerator
