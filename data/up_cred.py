# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.svm.libsvm_sparse import ConvergenceWarning

warnings.filterwarnings("ignore", category=DataConversionWarning, module="sklearn", lineno=761)
warnings.filterwarnings("ignore" ,category=ConvergenceWarning ,module="sklearn" ,lineno=931)
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


credit_cards=pd.read_csv('./creditcard.csv')

credit_cards["normAmount"] = StandardScaler().fit_transform(credit_cards["Amount"].values.reshape(-1,1))
del credit_cards["Amount"]
del credit_cards["Time"]

columns=credit_cards.columns
# The labels are in the last column ('Class'). Simply remove it to obtain features columns
features_columns=columns.delete(len(columns)-1)

features=credit_cards[features_columns]
labels=credit_cards['Class']

features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.2,
                                                                            random_state=0)

oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
len(os_labels[os_labels==1])

os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
c_param_range = [0.01, 0.1, 1, 10, 100]

content_list = []
for c_param in c_param_range:
    best_list = dict()
    print("-"*10)
    print("c_param_range" ,c_param)
    print("-"*10)
    lr = LogisticRegression(C = c_param,penalty="l1",solver='liblinear')
    lr.fit(os_features,os_labels)
    y_pred_undersample = lr.predict(features_test)
    print("逻辑回归准确率：", lr.score(features_test, labels_test))
    print("召回率", classification_report(labels_test, y_pred_undersample, labels=[0, 1]))
    best_list["flag"] = c_param
    best_list["num"] = lr.score(features_test, labels_test)
    content_list.append(best_list)
    best = list((max(content_list, key=lambda x: x["num"])).values())[0]

print("*"*100)
print(best)
lr = LogisticRegression(C = best, penalty = 'l1')
lr.fit(os_features,os_labels)
y_pred = lr.predict(features_test)
print("逻辑回归准确率：", lr.score(features_test, labels_test))
print("召回率", classification_report(labels_test, y_pred_undersample, labels=[0, 1]))

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(labels_test,y_pred)
# np.set_printoptions(precision=2)
#
# print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
# # Plot non-normalized confusion matrix
# class_names = [0,1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
# plt.show()

