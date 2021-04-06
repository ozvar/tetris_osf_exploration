import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# configure pandas table display
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


def sns_styleset():
    """Configure parameters for plotting"""
    sns.set_theme(context='paper',
                  style='whitegrid',
                  # palette='deep',
                  palette=['#c44e52', '#8c8c8c', '#937860', '#ccb974', '#4c72b0', '#dd8452'],
                  font='Arial')
    matplotlib.rcParams['figure.dpi']        = 300
    matplotlib.rcParams['axes.linewidth']    = 1
    matplotlib.rcParams['grid.color']        = '.8'
    matplotlib.rcParams['axes.edgecolor']    = '.15'
    matplotlib.rcParams['xtick.bottom']      = True
    matplotlib.rcParams['ytick.left']        = True
    matplotlib.rcParams['xtick.major.width'] = 1
    matplotlib.rcParams['ytick.major.width'] = 1
    matplotlib.rcParams['xtick.color']       = '.15'
    matplotlib.rcParams['ytick.color']       = '.15'
    matplotlib.rcParams['xtick.major.size']  = 3
    matplotlib.rcParams['ytick.major.size']  = 3
    matplotlib.rcParams['font.size']         = 11
    matplotlib.rcParams['axes.titlesize']    = 11
    matplotlib.rcParams['axes.labelsize']    = 10
    matplotlib.rcParams['legend.fontsize']   = 10
    matplotlib.rcParams['legend.frameon']    = False
    matplotlib.rcParams['xtick.labelsize']   = 10
    matplotlib.rcParams['ytick.labelsize']   = 10


def scree_plot(eigenvalues, xlabel=None, ylabel=None, hline=True):
    """Generate line plot of the eigenvalues obtained from factor analysis"""
    x_values = np.arange(1, len(eigenvalues)+1)
    plt.scatter(x_values, eigenvalues)
    plt.plot(x_values, eigenvalues)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if hline is True:
        plt.axhline(y=1, linestyle='--')
    plt.show()
    