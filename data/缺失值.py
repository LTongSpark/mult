# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

data =np.array([[1, 2], [np.nan, 3], [7, 6]])

imp = Imputer(missing_values='NaN' ,strategy='mean' ,axis=0)
print(imp.fit_transform(data))