import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def gaussion_1d(x, amp, cen, norm):
    """
    Calculate 1d gaussian function for given x

    param: x
    param: amp
    param: cen
    param: norm

    return: function
    """

    return amp * np.exp(-(x - cen) ** 2 / (2 * norm ** 2))


def gaussian_2d(x, y, amp, cen, norm):
    """
    Calculate 2d gaussian function for given x,y

    param: x - 1d np array
    param: y - 1d np array
    param: amp - 2d list
    param: cen - 2d list
    param: norm - 2d list

    return: scalar function
    """
    x, y = np.meshgrid(x, y)
    return gaussion_1d(x, amp[0], cen[0], norm[0]) * gaussion_1d(y, amp[1], cen[1], norm[1])


def inspector(matrix):
    """"
    Make simple 2D plot without any extras

    param: matrix - 2d np array
    """
    fig, axes = plt.subplots()
    im = axes.imshow(matrix, cmap="viridis")
    fig.colorbar(im)
    fig.show()


def _example():
    m = gaussian_2d(np.linspace(-20, 20, 100),
                    np.linspace(-20, 20, 100),
                    np.array([1,1]),
                    np.array([10,10]),
                    np.array([1,1]))

    inspector(m)


if __name__ == '__main__':
    _example()