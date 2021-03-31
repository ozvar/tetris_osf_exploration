import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# pandas tabulation options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# parameters for plotting theme
sns.set_theme(
    context='paper',
    style='whitegrid',
    palette=['#c44e52', '#8c8c8c', '#937860', '#ccb974', '#4c72b0', '#dd8452'],  # reordered version of seaborn 'deep' palette
    font_scale=1.5)
matplotlib.rcParams['figure.dpi'] = 200

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