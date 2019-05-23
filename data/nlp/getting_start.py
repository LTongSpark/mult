# -*- coding: utf-8 -*-
from nltk import word_tokenize, pos_tag
from nltk.corpus import brown
sens = "Mtachine learning is the science of getting computers to act without being explicitly programmed"
print(word_tokenize(sens))
print(pos_tag(word_tokenize(sens)))