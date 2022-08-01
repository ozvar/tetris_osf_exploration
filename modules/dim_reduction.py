import numpy as np
import pandas as pd
import pickle as pk

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


def factor_analyse(df, n_factors=3, rotation=None, method='minres', cutoff=None, display_loadings=False):
    """Conduct factor analysis of df using FactorAnalyzer and specified parameters, returning eigenvalues and factor loadings"""
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
    fa.fit(df)
    
    # obtain eigenvalues
    ev = fa.get_eigenvalues()[1]
    
    # create dataframe of factor loadings
    headers = [str(i) for i in np.arange(1, n_factors+1)]
    loadings = pd.DataFrame(fa.loadings_, 
                            index=df.columns, 
                            columns=headers)
    
    # create sorted dataframe of loadings above a certain cutoff (shows all by default) 
    if cutoff != None:
        loadings = loadings.where(abs(loadings).gt(cutoff))
        loadings = loadings.sort_values(by=headers, ascending=False)
        loadings = loadings.replace(np.nan, '', regex=True)
    
    if display_loadings is True:
        display(loadings)
        
    return ev, loadings, fa


def princomp(df, n_components=3, cutoff=None, display_loadings=False):
    """Conduct principal components analysis of df using sklearn's PCA, returning eigenvalues and factor loadings"""
    # Standardise the data
    df = pd.DataFrame(df, 
                      index=np.arange(0, len(df)), 
                      columns=df.columns)

    pca = PCA(n_components=n_components)
    pca.fit(df)
    
    # generate loadings table
    headers = [str(i) for i in np.arange(1, n_components+1)]
    loadings = pd.DataFrame(pca.components_.T, 
                            index=df.columns, 
                            columns=headers)
    
    # create sorted dataframe of loadings above a certain cutoff (shows all by default) 
    if cutoff != None:
        loadings = loadings.where(abs(loadings).gt(cutoff))
        loadings = loadings.sort_values(by=headers, ascending=False)
        loadings = loadings.replace(np.nan, '', regex=True)
    
    if display_loadings is True:
        display(loadings)
    
    # write pca for loading reloading in other analyses
    pk.dump(pca, open("pca.pkl","wb"))
    
    return loadings, pca
