import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale


data = pd.read_csv('./wine.data', header=None)
classes = data[0]
signs = data.iloc[:, 1:]
signs_scaled = scale(signs.values)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

score = []
score_scale = []
for x in range(1, 51):
    val_score = cross_val_score(KNeighborsClassifier(n_neighbors=x), X=signs.values, y=classes.values, cv=kf.split(data.values))
    val_score_scaled = cross_val_score(KNeighborsClassifier(n_neighbors=x), X=signs_scaled, y=classes.values, cv=kf.split(data.values))
    score.append(pd.Series(val_score).mean())
    score_scale.append(pd.Series(val_score_scaled).mean())

score_s = pd.Series(score)
score_scale_s = pd.Series(score_scale)
print('Без масштабирования: {} {}'.format(score_s.idxmax(), round(score_s.max(), 2)))
print('C масштабированием: {} {}'.format(score_scale_s.idxmax(), round(score_scale_s.max(), 2)))
