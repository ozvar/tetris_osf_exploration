import pandas as pd
import numpy as np


def compute_means(df, v, n):
    """Return an array containing the mean of variable 'var' at each match for all subject IDs and games in dataframe 'df' over 'n' episodes"""
    games = df.groupby(['SID', 'game_number'], sort=False).head(n)
    means = games.groupby('episode_number', sort=False)[v].mean().tolist()

    return means