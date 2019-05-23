# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

housing = fetch_california_housing()

data_train, data_test, target_train, target_test = train_test_split(housing.data, housing.target, test_size = 0.8)
dtr = tree.DecisionTreeRegressor(max_depth = 5)
dtr.fit(data_train, target_train)

# tree_param_grid = { 'min_samples_split': list((3,6,9)),'n_estimators':list((10,50,100))}
# grid = GridSearchCV(RandomForestRegressor(),param_grid=tree_param_grid, cv=5)
# grid.fit(data_train, target_train)
# print(grid.score(data_test,target_test))
# print(grid.best_score_)
# print(grid.best_params_)
result = RandomForestRegressor(min_impurity_split=6,n_estimators=100).min_impurity_decrease
print(result)