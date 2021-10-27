import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection
import seaborn as sns
from munging import compute_means, compute_errors


# configure pandas table display
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


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
    
    
def error_line(df, var, n, ax=None, xlabel=None, ylabel=None, label=None):
    """Plot the mean of variable 'var' at each match for all account_ids over 'n' matches, with shaded regions indicating SEM"""
    means = compute_means(df, var, n)
    errors = compute_errors(df, var, n)

    xticks = [1]
    xticks.extend(list(range(n // 10,
                             n + n // 10,
                             n // 10 if n // 10 != 0 else 1)))

    ax.plot(np.arange(1, n + 1), means)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.fill_between(np.arange(1, n + 1),
                     [means[i] + errors[i]*1.96 for i in range(n)],
                     [means[i] - errors[i]*1.96 for i in range(n)],
                     alpha=0.15)
 

class SeabornFig2Grid():
    """Enables seaborn multiplots"""
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

        
def viz_states(df, n_states, factors, factor_labels, post_prob, player_id, nth_game, cmap='tab10', fig_dir=None):
    """Create joint plot of latent state probabilities and observed states across tetris episodes"""
    game = df[(df['SID'] == player_id) & (df['game_number'] == nth_game)]
    episodes = len(game)
    state_colours = matplotlib.cm.get_cmap(cmap)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    states_legend = []
    for state in range(0, n_states):
        ax1.plot(np.arange(1, episodes), 
                 post_prob[:episodes-1, state],
                 color=state_colours(state))
        states_legend.append(f'State {state+1}')

    ax1.set_ylabel('Probability of state')
    ax1.legend(states_legend, 
               loc='upper left', 
               bbox_to_anchor=(1.05, 1), 
               frameon=True, 
               prop={'size': 8})

    factors_legend = [factor_labels[factor] for factor in factors]
    for factor in factors:
        ax2.plot(np.arange(1, episodes), game[factor][:episodes-1])

    ax2.set_ylabel('Factor score')
    ax2.set_xlabel('Episode')
    ax2.legend(factors_legend, 
               loc='upper left', 
               bbox_to_anchor=(1.05, 1), 
               frameon=True, 
               prop={'size': 8})

    fig.suptitle(f'Player {player_id} Game {nth_game}: {n_states}-State Model')
    if fig_dir:
        fig_name = f'{n_states}_state_HMM_player_{player_id}_game_{nth_game}'
        plt.savefig(os.path.join(fig_dir, fig_name), bbox_inches='tight')

    plt.show()
