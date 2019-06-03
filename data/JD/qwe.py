# -*- coding: utf-8 -*-

import seaborn as sns
def read_item_names():


    file_name = ('D:/大数据/python/python2/14-人工智能阶段：-机器学习-深度学习-实战项目/26-28机器学习算法配套案例实战/推荐系统/推荐系统/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid

print(read_item_names())
