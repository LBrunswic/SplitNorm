import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import markers
import DatasetManager.base as base

def BuildDataSet2D(TFdataset,vizKWarg={}):
    def plot_points(data, axis, s=1, color='b', label='', style=markers.MarkerStyle(marker='.')):
        x, y = np.squeeze(np.split(data, 2, axis=1))
        axis.scatter(x, y, c=color,marker=style)

    def plot_family(samples, limits=True):

        fig, ax = plt.subplots(
            1, 1)

        colors = ['b','g','r','c','m','y','salmon','lime','purple','yellow','lightgreen','gray','black','orange','navy']
        for i in range(len(samples)):
            plot_points(samples[i], ax, s=30, color=colors[i], label='ode(samples)')
        if limits:
            set_limits([ax], -2, 3.5, -3, 2)
        return fig
    def set_limits(axes, min_x, max_x, min_y, max_y):
      if isinstance(axes, list):
        for axis in axes:
          set_limits(axis, min_x, max_x, min_y, max_y)
      else:
        axes.set_xlim(min_x, max_x)
        axes.set_ylim(min_y, max_y)


    def picture_gen(sample):
        return plot_family(sample)
    visualization = base.Visualization(picture_gen,**vizKWarg)

    return base.Dataset(TFdataset,visualization)


