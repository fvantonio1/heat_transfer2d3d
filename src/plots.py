
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

def plot_temperatura_pico(fig, tm, plot_points=False, title="Temperatura de Pico (°C)"):
    x, y = tm[:,0], tm[:,1]
    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    s_grid = griddata((x, y), tm[:,2], (X, Y), method='nearest')
    X = np.concatenate((X[:, ::-1]*-1, X), axis=1)
    Y = np.concatenate((Y, Y), axis=1)
    s_grid = np.concatenate((s_grid[:, ::-1], s_grid), axis=1)

    if plot_points:
        plt.plot(X, Y, marker='.', color='k', linestyle='none', markersize=2)

    cs = plt.contourf(X, Y, s_grid, levels=50, alpha=.99, cmap='jet')
    #cs2 = plt.contour(cs, Y, s_grid, 5, colors='black')
    #plt.clabel(cs2, inline=1, fontsize=10)
    plt.colorbar(label='Temperatura (°C)')
    plt.xlabel('Largura (m)')
    plt.ylabel('Espessura (m)')
    plt.title(title, fontsize=20)
    #plt.axis('equal')
    return fig