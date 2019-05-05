import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston

data = load_boston()
data_scaled = scale(data.data)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
score = []

for x in np.linspace(1, 10, 200):
    val_score = cross_val_score(KNeighborsRegressor(n_neighbors=5, weights='distance', p=x),
                                scoring='neg_mean_squared_error', X=data_scaled, y=data.target, cv=kf.split(data.data))
    score.append(pd.Series(val_score).mean())

p = pd.Series(score).idxmax() + 1
print("Optimal p: {}".format(p))
