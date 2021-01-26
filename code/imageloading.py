'''

aim: create a function which would load the .mat file 
date: 25.01.2021
author: Alina

'''
import scipy.io as sio
import matplotlib.pyplot as plt

def load_image(path):
    """Load the image and display it 

    Parameters
    ----------
    path : string
        path to the file

    Returns
    -------
    nothing
    displays figure
    """
    Im = sio.loadmat(path)
    

path = '2dpeakfinder/data/brain.mat'
Im = load_image(path)
plt.imshow(Im['Im'], cmap = 'gray')

