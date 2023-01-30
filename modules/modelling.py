import os
import pickle as pk
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
    - returns model, state probabilities, and input array, printing transition matrix and descriptive metrics"""
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
    
    if verbose:
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
        
        print('---------------------------\n'
              'Fractional occupancy for each state:\n'
              '---------------------------\n')      
        print(fractional_occupancy(model, X, n_states, factors).to_string(header=False), '\n')
        
        print(f'Switch rate of model is {switch_rate(model, X)}\n')

        print(f'Log-likelihood of model is {LL}')
    
    return model, post_prob, X, LL


def fit_group_HMM(df, 
                  n_states, 
                  components, 
                  component_labels, 
                  n_iter=200, 
                  verbose=True, 
                  covar_type='diag', 
                  null_model=False,
                  nth_fit=1,
                  model_dir=None):
    """Fit HMM, using arrays of components as observed states, to all tetris
    games of all players in data set - returns model, state probabilities,
    and input array, printing transition matrix and descriptive metrics"""
    # instantiate model
    start_prob = rand_start_prob(n_states)
    model = hmm.GaussianHMM(n_components = n_states,
                        covariance_type=covar_type,
                        n_iter=n_iter)
    # structure data 
    component_arrays = [np.array(df[component]) for component in components]
    # reshuffle data if null model is of interest
    if null_model:
        component_arrays = [np.random.choice(array, len(array), replace=False)
                            for array in component_arrays]
    # reshape data for fitting
    X = np.column_stack(component_arrays)
    # fit model
    model.fit(X)
    post_prob = model.predict_proba(X)
    
    LL = np.round(model.score(X), 2)
    
    if verbose:
        if not null_model:
            print(f'Fitting {n_states} state model to all games of all players')
        else:
            print(f'Fitting {n_states} state NULL MODEL to all games of all players')

        print('---------------------------\n'
              'Transition probabilities:\n'
              '---------------------------')
        print(tabulate_trans_probs(model, n_states), '\n')

        print('---------------------------\n'
              'Component means for each state:\n'
              '---------------------------\n')
        print(tabulate_means(model, components, component_labels, n_states), '\n')

        print(f'Switch rate of model is {switch_rate(model, X)}\n')

        print(f'Log-likelihood of model is {LL}')
    
    if model_dir != None:
        data = [model, post_prob, X, LL, start_prob]
        save_group_HMM(
                data,
                n_states,
                components,
                null_model,
                nth_fit,
                model_dir)

    return model, post_prob, X, LL


def save_group_HMM(
        data,
        n_states,
        components,
        null_model,
        nth_fit,
        model_dir):
        
        
        n_comps = len(components)
        if null_model:
            pickle_name = f'{n_states}_state_{n_comps}_components_chance_model_fit_{nth_fit}.pkl'
            model_dir = os.path.join(
                    model_dir,
                    f'{n_states}_state_{n_comps}_component_chance_model'
                    )
        else:
            pickle_name = f'{n_states}_state_{n_comps}_component_group_HMM_fit_{nth_fit}.pkl'
            model_dir = os.path.join(
                    model_dir,
                    f'{n_states}_state_{n_comps}_component_HMM'
                    )

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        else:
            with open (os.path.join(model_dir, pickle_name), "wb") as f:
                pk.dump(len(data), f)
                for datum in data:
                    pk.dump(datum, f)


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


def fractional_occupancy(model, observed_states, n_states, factors):
    '''Calculate for each state the fraction of total occurrences occupied by that state, return all fractions as a vector'''
    state_occurrences = list(model.predict(observed_states)) 
    fractional_occupancies = []
    for i in range(n_states):
        fractional_occupancy = np.round(state_occurrences.count(i) / len(state_occurrences),
                                        decimals=4)
        fractional_occupancies.append(fractional_occupancy)

    row_names = {int(i): f'State {i+1}' for i in range(n_states)}
    fo_df = pd.DataFrame({'Fractional occupancy': fractional_occupancies})
    fo_df.rename(index=row_names, inplace=True)

    return fo_df


def switch_rate(model, observed_states):
    '''Calculate rate of switches between states, calculated as number of switches divided by total state occurrences'''
    states = model.predict(observed_states)
    switch_count = np.count_nonzero(np.diff(states))

    rate = switch_count / len(states - 1)
    rate = np.round(rate, 4)
    
    return rate
