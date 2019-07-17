# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas_profiling

sns.set(color_codes=True)
sns.set_style("whitegrid")
np.random.seed(sum(map(ord,'regression')))
tips = sns.load_dataset('tips')
data =  pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

if __name__ == '__main__':
    prf = pandas_profiling.ProfileReport(data)
    prf.to_file(('./ex.html'))

