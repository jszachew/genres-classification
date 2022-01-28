from __future__ import print_function

from statistics import mean

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.utils import class_weight
from imblearn.ensemble import BalancedRandomForestClassifier

from src.data_preprocessing import prepare_arrays_of_occurance_test_cd1, insert_cd1_test_x_to_array, prepare_test_y_cd1, \
    prepare_arrays_of_occurance_train_cd1, insert_cd1_train_x_to_array, prepare_train_y_cd1, \
    prepare_arrays_of_occurance_train_cd2, insert_cd2_train_x_to_array, prepare_train_y_cd2
from utils import plot_confusion_matrix


def train_cd2_random_forest(weighted=True):

    arr_train = prepare_arrays_of_occurance_train_cd2()
    train_x = insert_cd2_train_x_to_array(arr_train)
    train_y = prepare_train_y_cd2()
    pd_train_y = pd.array(train_y)
    pd_train_x = pd.array(train_x, dtype=int)


    x_train, x_test, y_train, y_test = train_test_split(pd_train_x, pd_train_y, test_size=0.2, stratify=pd_train_y)

    x_train = preprocessing.normalize(x_train)
    x_train = preprocessing.scale(x_train)

    x_test = preprocessing.normalize(x_test)
    x_test = preprocessing.scale(x_test)

    print(Counter(y_train).keys())
    print(Counter(y_train).values())

    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    print(weights)

    print("x_train: %s, x_test: %s, y_train: %s, y_test: %s" % (len(x_train), len(x_test), len(y_train), len(y_test)))

    print("RandomForestClassifier")
    if weighted:
        lr = BalancedRandomForestClassifier(n_estimators=150, verbose=1, random_state=2, n_jobs=-1)
    else:
        lr = RandomForestClassifier(n_estimators=100,max_features="sqrt", verbose=1, n_jobs=-1)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scoring = ('f1_macro', 'recall_macro', 'precision_macro')
    # Evaluate BRFC model
    scores = cross_validate(lr, pd_train_x, pd_train_y, scoring=scoring, cv=cv)
    # Get average evaluation metrics
    print('Mean f1: %.3f' % mean(scores['test_f1_macro']))
    print('Mean recall: %.3f' % mean(scores['test_recall_macro']))
    print(scores['test_recall_macro'])
    print(scores['test_precision_macro'])
    print('Mean precision: %.3f' % mean(scores['test_precision_macro']))
    lr.fit(x_train, y_train)
    print(f"accuracy for BRandomForestClassifier: {(lr.score(x_test, y_test))}")

    y_pred = lr.predict(x_test)

    # Accuracy
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')
    # plot_confusion_matrix(cm=cm, classes=classes, normalize=True)
    print(cm)
    plot_confusion_matrix(cm=cm, classes=classes, title=f"CD2 BRandomForestClassifier imbalanced 100 sqrt")
    print(classification_report(y_test, y_pred, target_names=classes))
    print(accuracy)
    # print(f"f1: {f1}")


if __name__ == '__main__':
    train_cd2_random_forest(weighted=False)
