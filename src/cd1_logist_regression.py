from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from collections import Counter
from sklearn.utils import class_weight

from src.data_preprocessing import prepare_arrays_of_occurance_test_cd1, insert_cd1_test_x_to_array, prepare_test_y_cd1, \
    prepare_arrays_of_occurance_train_cd1, insert_cd1_train_x_to_array, prepare_train_y_cd1, \
    prepare_arrays_of_occurance_train_cd2, insert_cd2_train_x_to_array, prepare_train_y_cd2
from utils import plot_confusion_matrix


def train_cd1_logistic_regression(weighted=True):
    w = {
        "RnB": 2.21383523,
        "Country": 1.63764602,
        "Pop_Rock": 0.10347397,
        "New Age": 35.72115385,
        "Latin": 3.39864298,
        "Folk": 3.44646308,
        "Rap": 1.85464076,
        "Jazz": 5.3248925,
        "Electronic": 2.00188603,
        "Reggae": 4.62160481,
        "Blues": 7.97495528,
        "International": 14.65483235,
        "Vocal": 40.82417582,
    }

    arr_train = prepare_arrays_of_occurance_train_cd1()
    train_x = insert_cd1_train_x_to_array(arr_train)
    train_y = prepare_train_y_cd1()
    pd_train_y = pd.array(train_y)
    pd_train_x = pd.array(train_x, dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(pd_train_x, pd_train_y, test_size=0.2)
    x_train = preprocessing.normalize(x_train)
    x_train = preprocessing.scale(x_train)

    x_test = preprocessing.normalize(x_test)
    x_test = preprocessing.scale(x_test)

    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    print("x_train: %s, x_test: %s, y_train: %s, y_test: %s" % (len(x_train), len(x_test), len(y_train), len(y_test)))

    print("Logistic Regression Classifier")
    if weighted:
        lr = LogisticRegression(solver="sag", multi_class="ovr", verbose=1, n_jobs=-1, class_weight=w)
    else:
        lr = LogisticRegression(solver="sag", multi_class="ovr", verbose=1, n_jobs=-1)
    lr.fit(x_train, y_train)
    print(f"accuracy for LogisticRegression: {(lr.score(x_test, y_test))}")

    y_pred = lr.predict(x_test)

    # Accuracy
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')
    plot_confusion_matrix(cm=cm, classes=classes, title=f"CD1 LogisticRegression sag ovr")
    print(classification_report(y_test, y_pred, target_names=classes))
    print(accuracy)


if __name__ == '__main__':
    train_cd1_logistic_regression(weighted=False)
