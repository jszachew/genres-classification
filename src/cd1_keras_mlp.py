from __future__ import print_function
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten, Dropout, Activation

from src.data_preprocessing import prepare_arrays_of_occurance_test_cd1, insert_cd1_test_x_to_array, prepare_test_y_cd1, \
    prepare_arrays_of_occurance_train_cd1, insert_cd1_train_x_to_array, prepare_train_y_cd1, \
    prepare_arrays_of_occurance_train_cd2, insert_cd2_train_x_to_array, prepare_train_y_cd2
from utils import plot_confusion_matrix, plot_history
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model

import matplotlib.pyplot as plt


def train_cd1_mlp(weighted=True):
    from keras.backend import clear_session
    clear_session()
    arr_train = prepare_arrays_of_occurance_train_cd1()
    train_x = insert_cd1_train_x_to_array(arr_train)
    train_y = prepare_train_y_cd1()
    pd_train_y = pd.array(train_y)
    pd_train_x = pd.array(train_x, dtype=int)

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
    input_dim = x_train.shape[1]  # Number of features
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
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), normalize='true')
    print(cm)
    plot_confusion_matrix(cm=cm, classes=classes, title=f"CD1 MLP with dropout sgd network")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=classes))


if __name__ == '__main__':
    train_cd1_mlp(weighted=True)