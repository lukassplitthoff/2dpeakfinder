import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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


def gradientmethode(matrix):

    g = np.gradient(matrix, dtype=float)

    return g


def _example():
    d = np.loadtxt('../data/2021-01-26_multiple-peaks.txt')

    g = gradientmethode(d)
    print(type(g))
    plottr(d,
           points=np.array([[100, 100], [200, 200]]),
           #points=np.array([[100,100], [200,200]]),
           save="../images/2021-01-26_multiple-peak-plot.png")

    #plottr(g,
     #      points=np.array([]),
     #      save="../images/2021-01-26_multiple-peak-plot.png")


if __name__ == '__main__':
    _example()