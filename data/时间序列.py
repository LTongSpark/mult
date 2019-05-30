# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

rng = pd.date_range('2016-07-01',periods=10,freq='3d')
print(rng)

time=pd.Series(np.random.randn(20),index=pd.date_range('2016-09-08',periods=20))
print(time)
#往后取
print(time.truncate(before='2016-09-12'))
#往前取
print(time.truncate(after='2016-09-12'))

