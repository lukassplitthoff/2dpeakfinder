import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from peakgenerator import gaussian_2d
import scipy.optimize as opt

def plottr(matrix, points, save):
    """
    Calculate 2d gaussian function for given x and y

    Parameter
    ---------
    matrix: float, array
    points: float, array
    save: str

    Returns
    -------
    figure
    """
    fig, axes = plt.subplots()
    im = axes.imshow(matrix, cmap="viridis")
    for i in range(len(points)):
        axes.scatter(points[i,0], points[i,1], s=2, c='red', marker='o')
    fig.colorbar(im)
    fig.show()
    if type(save) == str:
        fig.savefig(save)
    pass


def find_local_maxima(matrix, threshold):
    """ in the 2D array Im find local max
    Parameters
    ----------
    Im : 2D numpy array
    Returns
    -------
    yx : numpy array
        coordinates of local max
    """
    pnt_max = peak_local_max(matrix, min_distance=1, threshold_abs=threshold)

    return pnt_max


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = opt.leastsq(errorfunction, params)
    return p


def peakfitter(data, threshold):
    """"
    Wraps around find_local_maxima and fit gaussion to first finds the peaks and then extract the peak values

    Parameters
    ----------
    data: float, 2d array
    threshold: float


    Returns
    -------
    pnt_max: float, array of guessed maximum position
    results: float, array of fit parameters

    """

    pnt_max = find_local_maxima(data.T, threshold=threshold)

    results = np.zeros((len(pnt_max), 5), dtype=object)

    for i in range(len(pnt_max)):

        cutout = 20
        ymax = pnt_max[i, 0]
        xmax = pnt_max[i, 1]

        a = xmax -20
        b = xmax +20
        c = ymax -20
        d = ymax +20

        if xmax-cutout <= 0:
            a = xmax
        elif ymax-cutout <= 0:
            c = xmax
        elif xmax+cutout >= 999:
            b = 999
        elif ymax+cutout >= 999:
            d = 999

        data_select = data[a:b,c:d]

        params = fitgaussian(data_select)
        results[i] = params
        results[i, 1:3] += np.array([a,c])

    return pnt_max, results


def _example():
    d = 0
    for i in range(20):
        d = d + gaussian_2d(np.linspace(0, 100, 1000),
                            np.linspace(0, 100, 1000),
                            np.random.random(2),
                            np.random.randint(low=0, high=100, size=(1,2))[0],
                            np.random.randint(low=1, high=20, size=(1,2))[0]/10
                            )

    pnts, results = peakfitter(d, threshold=0.001)
    plottr(d, points=pnts, save="../images/2021-01-26_multiple-peak-plot.png")
    print(results, pnts)
    print(len(pnts))



if __name__ == '__main__':
    _example()