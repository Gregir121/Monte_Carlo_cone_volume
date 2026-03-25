import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def numpy_integral(n_max, radius, height):


    x_step = np.random.uniform(-radius, radius, size = n_max)
    y_step = np.random.uniform(-radius, radius, size = n_max)
    z_step = np.random.uniform(0, height, size = n_max)

    moving_radius = (radius * ((height - z_step) / height))

    inside = (x_step**2 + y_step**2 <= moving_radius**2 )

    hit = np.cumsum(inside)

    ens = np.arange(1, n_max + 1)

    box_volume = (2 * radius) ** 2 * height

    cone_volume = box_volume * (hit / ens)

    df_points = pd.DataFrame({
        'x': x_step,
        'y': y_step,
        'z': z_step,
        'inside': inside
    })


    return cone_volume, df_points


if __name__ == "__main__":

    colors_theme = ['olivedrab', 'sandybrown', 'navy', 'goldenrod','blueviolet', 'darkorchid', 'aqua', 'black', 'orchid', 'lightgreen']

    theo_volume = (1/3 * np.pi * 4**2 * 8 )

    fig, ax = plt.subplots(1,1, figsize = (12, 10))
    ax.set_xlim(0, 100001)
    ax.set_title('Convergence of the Monte Carlo method depending on the number of samples')
    ax.set_ylim(theo_volume * 0.8, theo_volume * 1.2)
    ax.grid(True, alpha = 0.3)

    last_df = None
    results_100 = []
    results_1000 = []
    results_10000 = []
    results_100000 = []



    for i in range(10):
        list_of_res, points = numpy_integral(100000, 4, 8)
        last_df = points
        results_100.append(list_of_res[99])
        results_1000.append(list_of_res[999])
        results_10000.append(list_of_res[9999])
        results_100000.append(list_of_res[99999])
        ax.plot(np.arange(1, 100001), list_of_res, color = colors_theme[i], linewidth = 1, alpha = 0.8)
        plt.draw()
        plt.pause(0.1)
        fig.savefig('graphs/convergence_plot.png')

    results_list = [results_100, results_1000, results_10000,results_100000]



    points_list = [
        last_df.iloc[:100],
        last_df.iloc[:1000],
        last_df.iloc[:10000],
        last_df.iloc[:100000]
    ]

    for item in points_list:
        H = 8
        r_max = 4

        fig_3D, ax_3D = plt.subplots(subplot_kw={'projection': '3d'})
        r = np.linspace(0, r_max, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        R, Theta = np.meshgrid(r, theta)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        Z = H - (H / r_max) * R
        ax_3D.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, rstride=10, cstride=10)

        color_green = to_rgba('green', alpha=1.0)
        color_red = to_rgba('red', alpha=0.09)

        colors = item['inside'].map({True: color_green, False: color_red})

        ax_3D.scatter(item['x'], item['y'], item['z'], c=colors)
        ax_3D.set_box_aspect([1, 1, 1])
        ax_3D.set_title(f'Hit chart for N = {len(item)}')
        ax_3D.set_xlabel('x')
        ax_3D.set_ylabel('y')
        ax_3D.set_zlabel('z')


        fig_3D.savefig(f'graphs/graph_{len(item)}_points.png')


    fig2, ax2 = plt.subplots(1, 1, figsize = (15,8))
    ax2.boxplot(results_list, patch_artist=True)
    ax2.set_title('Graph of convergence depending on the drawing series')
    ax2.set_xticklabels(['N=100', 'N=1000', 'N=10000', 'N=100000'])
    ax2.set_ylim(0, 200)
    ax2.axhline(y = theo_volume, linestyle = '--', color = 'red')
    fig2.savefig('graphs/boxplot_results.png')


    plt.show()












