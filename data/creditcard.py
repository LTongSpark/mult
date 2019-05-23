# -*- coding: utf-8 -*-
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm.libsvm_sparse import ConvergenceWarning


class creditcard(object):
    def __init__(self):
        self.data = pd.read_csv("./creditcard.csv")
    def run(self):
        warnings.filterwarnings("ignore", category=DataConversionWarning, module="sklearn", lineno=761)
        warnings.filterwarnings("ignore" ,category=ConvergenceWarning ,module="sklearn" ,lineno=931)
        # count_classes = pd.value_counts(self.data['Class'] ,sort=True).sort_index()
        # count_classes.plot(kind = 'bar')
        # plt.title("Fraud class histogram")
        # plt.xlabel("Class")
        # plt.ylabel("Frequency")

        self.data["normAmount"] = StandardScaler().fit_transform(self.data["Amount"].values.reshape(-1,1))
        del self.data["Amount"]
        del self.data["Time"]

        x = self.data.loc[:,self.data.columns != 'Class']
        y = self.data.loc[:,'Class']
        number_records_fraud = len(self.data[self.data.Class == 1])
        fraud_indices = np.array(self.data[self.data.Class == 1].index)
        # Picking the indices of the normal classes
        normal_indices = self.data[self.data.Class == 0].index

        # Out of the indices we picked, randomly select "x" number (number_records_fraud)
        random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
        random_normal_indices = np.array(random_normal_indices)

        # Appending the 2 indices
        under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

        # Under sample dataset
        under_sample_data = self.data.iloc[under_sample_indices, :]

        X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
        y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']
        '''
        Percentage of normal transactions:  0.5
        Percentage of fraud transactions:  0.5
        Total number of transactions in resampled data:  984
        '''
        # Showing ratio
        print("Percentage of normal transactions: ",
              len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
        print("Percentage of fraud transactions: ",
              len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
        print("Total number of transactions in resampled data: ", len(under_sample_data))

        '''
        Number transactions train dataset:  199364
        Number transactions test dataset:  85443
        Total number of transactions:  284807
        
        Number transactions train dataset:  688
        Number transactions test dataset:  296
        Total number of transactions:  984
        '''
        # Whole dataset
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        print("Number transactions train dataset: ", len(X_train))
        print("Number transactions test dataset: ", len(X_test))
        print("Total number of transactions: ", len(X_train) + len(X_test))

        # Undersampled dataset
        X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(
            X_undersample
            , y_undersample
            , test_size=0.3
            , random_state=0)
        print("")
        print("Number transactions train dataset: ", len(X_train_undersample))
        print("Number transactions test dataset: ", len(X_test_undersample))
        print("Total number of transactions: ", len(X_train_undersample) + len(X_test_undersample))

        c_param_range = [0.01, 0.1, 1, 10, 100]

        content_list = []
        for c_param in c_param_range:
            best_list = dict()
            print("-"*10)
            print("c_param_range" ,c_param)
            print("-"*10)
            lr = LogisticRegression(C = c_param,penalty="l1",solver='liblinear')
            lr.fit(X_train_undersample,y_train_undersample)
            y_pred_undersample = lr.predict(X_test_undersample)

            #print("逻辑回归准确率：", lr.score(X_test_undersample, y_test_undersample))
            #print("召回率", classification_report(y_test_undersample, y_pred_undersample, labels=[0, 1]))
            best_list["flag"] = c_param
            best_list["num"] = lr.score(X_test_undersample, y_test_undersample)
            content_list.append(best_list)

            best = list((max(content_list,key=lambda x:x["num"])).values())[0]

            import itertools
            lr = LogisticRegression(C=best, penalty='l1')
            lr.fit(X_train_undersample, y_train_undersample.values.ravel())
            y_pred_undersample = lr.predict(X_test_undersample.values)

            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
            np.set_printoptions(precision=2)

            print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

            # Plot non-normalized confusion matrix
            class_names = [0, 1]
            plt.figure()
            self.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
            plt.show()

        return best

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
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')



if __name__ == '__main__':
    print(creditcard().run())