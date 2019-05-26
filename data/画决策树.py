# -*- coding: utf-8 -*-
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz


def print_graph(clf,feature_names):
    '''
    print decision tree
    :param clf:
    :param feature_names:
    :return:
    '''
    graph = export_graphviz(
        clf,
        label='root',
        proportion=True,
        impurity=False,
        out_file=None,
        feature_names=feature_names,
        class_names={0:'D',1:'R'},
        filled=True,
        rounded=True
    )
    graph = pydotplus.graph_from_dot_data(graph)
    return Image(graph.create_png())