# -*- coding: utf-8 -*-
from nltk.corpus import names

import seaborn as sn

sn.heatmap()
import random
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
print(len(names))
print(names[0:10])