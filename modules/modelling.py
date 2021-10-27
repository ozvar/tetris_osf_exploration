import pandas as pd
import numpy as np
import scipy.stats as scipy

from hmmlearn import hmm


def rand_start_prob(n_states):
    start_prob = np.random.rand(n_states)
    start_prob = start_prob / sum(start_prob)

    # subtract any imprecision from array to ensure it sums to 1
    diff = sum(start_prob) - 1
    start_prob[0] -= diff
    
    return start_prob


def fit_HMM(df, n_states, factors, factor_labels, player_id, nth_game, n_iter, verbose=True, covar_type='diag', null_model=False):
    """Fit HMM, using arrays of factors as observed states, to single tetris game of specified player
    - returns model, printing transition matrix and log-likelihood"""
    # instantiate model
    model = hmm.GaussianHMM(n_components = n_states,
                        covariance_type='diag',
                        n_iter=200)
    # structure data 
    game = df[(df['SID'] == player_id )
              & (df['game_number'] == nth_game)]
    # reshape arrays
    factor_arrays = [np.array(game[factor]) for factor in factors]
    # reshuffle data if null model is of interest
    if null_model:
        factor_arrays = [np.random.choice(array, len(array), replace=False) for array in factor_arrays]
    # reshape data for fitting
    X = np.column_stack(factor_arrays)
    # fit model
    model.fit(X)
    post_prob = model.predict_proba(X)
    
    LL = np.round(model.score(X), 2)
    
    if null_model is False:
        print(f'Fitting {n_states} state model to game {nth_game} of player {player_id}')
    print('---------------------------\n'
          'Transition probabilities:\n'
          '---------------------------')
    print(tabulate_trans_probs(model, n_states), '\n')
    
    print('---------------------------\n'
          'Component means for each state:\n'
          '---------------------------\n')
    print(tabulate_means(model, factors, factor_labels, n_states), '\n')

    print(f'Log-likelihood of model is {LL}')
    
    return model, post_prob, LL


def tabulate_means(model, factors, factor_labels, n_states):
    
    means = np.transpose(model.means_)
    means_df = pd.DataFrame(means)

    row_names = {int(i): factor_labels[factors[i]] for i in range(len(factors))}
    col_names = {int(i): f'State {i+1}' for i in range(n_states)}

    means_df.rename(index=row_names, inplace=True)
    means_df.rename(columns=col_names, inplace=True)

    return means_df


def tabulate_trans_probs(model, n_states):
    
    trans_mat = np.round(model.transmat_, 2)
    trans_df = pd.DataFrame(trans_mat)
    
    row_names = col_names = {int(i): f'State {i+1}' for i in range(n_states)}
    
    trans_df.rename(index=row_names, inplace=True)
    trans_df.rename(columns=col_names, inplace=True)
    
    return trans_df


def cross_corr(n, a, b):
    corr_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr = scipy.pearsonr(a[i], b[j])[0]
            corr_mat[i, j] = corr
            
    return corr_mat


def check_unique_state_matches(n, df, threshold):
    """Returns true if each row and column of cross-correlation matrix has single correlation value above threshold"""
    for i in range(n):
        check_row = sum(df.iloc[i, :] > threshold)
        if check_row != 1:
            print(f'**States failed to match**')
            return False
        check_col = sum(df.iloc[:, i] > threshold)
        if check_col != 1:
            print('States failed to match')
            return False
    
    print('Matching states identified')
    return True
