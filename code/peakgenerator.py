import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def gaussion_1d(x, amp, cen, norm):
    """
    Calculate 1d gaussian function for given x

    Parameter
    ---------
    x: float
    amp: float
    cen: float
    norm: float

    Returns
    -------
    function: float
    """

    return amp * np.exp(-(x - cen) ** 2 / (2 * norm ** 2))


def gaussian_2d(x, y, amp, cen, norm):
    """
    Calculate 2d gaussian function for given x and y

    Parameter
    ---------
    x: float
    y: float
    amp: float, list
    cen: float, list
    norm: float, list

    Returns
    -------
    function: float
    """
    x, y = np.meshgrid(x, y)
    return gaussion_1d(x, amp[0], cen[0], norm[0]) * gaussion_1d(y, amp[1], cen[1], norm[1])


def inspector(matrix, save):
    """
    Calculate 2d gaussian function for given x and y

    Parameter
    ---------
    matrix: float, array

    Returns
    -------
    figure
    """
    fig, axes = plt.subplots()
    im = axes.imshow(matrix, cmap="viridis")
    fig.colorbar(im)
    fig.show()
    if save == True:
        fig.savefig("../images/2021-01-26_peak.png")
    pass


def _example():
    m = gaussian_2d(np.linspace(-20, 20, 100),
                    np.linspace(-20, 20, 100),
                    np.array([1,1]),
                    np.array([10,10]),
                    np.array([1,1]))

    inspector(m, save=True)

    np.savetxt("../data/2021-01-26.txt", m)


if __name__ == '__main__':
    _example()