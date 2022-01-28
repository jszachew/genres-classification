from __future__ import print_function
import time
from statistics import mean
from keras import backend as K
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from collections import Counter
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding

from sklearn.neural_network import MLPClassifier
from sklearn.utils import class_weight
from imblearn.ensemble import BalancedRandomForestClassifier

from src.data_preprocessing import prepare_arrays_of_occurance_test_cd1, insert_cd1_test_x_to_array, prepare_test_y_cd1, \
    prepare_arrays_of_occurance_train_cd1, insert_cd1_train_x_to_array, prepare_train_y_cd1, \
    prepare_arrays_of_occurance_train_cd2, insert_cd2_train_x_to_array, prepare_train_y_cd2
from utils import plot_confusion_matrix, plot_history
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras import layers, optimizers

from numpy import array
from numpy import argmax
import matplotlib.pyplot as plt


def train_cd1_mlp(weighted=True):
    from keras.backend import clear_session
    clear_session()
    arr_train = prepare_arrays_of_occurance_train_cd1()
    train_x = insert_cd1_train_x_to_array(arr_train)
    train_y = prepare_train_y_cd1()
    pd_train_y = pd.array(train_y)
    pd_train_x = pd.array(train_x, dtype=int)
    print(pd_train_x[:3])


    oversample = SMOTE(k_neighbors=5)
    over_X, over_y = oversample.fit_resample(pd_train_x, pd_train_y)

    le = preprocessing.LabelEncoder()
    le.fit(over_y)
    print(over_y)
    over_y = le.transform(over_y)
    print("inverse")
    print(le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    over_y = to_categorical(over_y)

    x_train, x_test, y_train, y_test = train_test_split(over_X, over_y, test_size=0.2, stratify=over_y)

    classes = np.unique(le.inverse_transform(np.argmax(y_train, axis=1)))
    print(classes)
    input_dim = x_train.shape[1]  # Number of features

    #model = Sequential([
    #    Embedding(5000, 128),
    #    Conv1D(128, 5, activation='relu'),
    #    MaxPooling1D(5),
    #    Dropout(0.2),
    #    Conv1D(64, 5, activation='relu'),
    #    MaxPooling1D(5),
    #    Dropout(0.2),
    #    Conv1D(32, 5, activation='relu'),
    #    GlobalMaxPooling1D(),
    #    Dense(32, activation='relu'),
    #    Dense(13, activation='softmax'),
    #])
    batch_size = 128
    hidden_units = 256
    dropout = 0.45

    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(5000,)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(13, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    model.summary()
    print(y_train)
    history = model.fit(x_train, y_train,
                        epochs=400,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        batch_size=32)

    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)
    plt.clf()
    y_pred = model.predict(x_test)
    print(y_pred)
    print(np.argmax(y_pred, axis=1))
    print(np.argmax(y_test, axis=1))
    # Accuracy
    # from sklearn.metrics import accuracy_score, f1_score

    # accuracy = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), normalize='true')
    # plot_confusion_matrix(cm=cm, classes=classes, normalize=True)
    print(cm)
    plot_confusion_matrix(cm=cm, classes=classes, title=f"CD1 MLP with dropout sgd network")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=classes))
    # print(accuracy)
    # print(f"f1: {f1}")


if __name__ == '__main__':
    train_cd1_mlp(weighted=True)