from __future__ import print_function
import time
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding

from src.data_preprocessing import prepare_arrays_of_occurance_test_cd1, insert_cd1_test_x_to_array, prepare_test_y_cd1, \
    prepare_arrays_of_occurance_train_cd1, insert_cd1_train_x_to_array, prepare_train_y_cd1, \
    prepare_arrays_of_occurance_train_cd2, insert_cd2_train_x_to_array, prepare_train_y_cd2
from utils import plot_confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
import matplotlib.pyplot as plt


def plot_history(history):
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"MLP_350_50_13_history_{now}.png")
    plt.clf()

def train_cd1_mlp(weighted=True):
    from keras.backend import clear_session
    clear_session()
    arr_train = prepare_arrays_of_occurance_train_cd1()
    train_x = insert_cd1_train_x_to_array(arr_train)
    train_y = prepare_train_y_cd1()
    pd_train_y = pd.array(train_y)
    pd_train_x = pd.array(train_x, dtype=int)

    oversample = SMOTE(k_neighbors=1)
    over_X, over_y = oversample.fit_resample(pd_train_x, pd_train_y)

    le = preprocessing.LabelEncoder()
    le.fit(over_y)
    over_y = le.transform(over_y)
    print("inverse")
    print(le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    over_y = to_categorical(over_y)

    x_train, x_test, y_train, y_test = train_test_split(over_X, over_y, test_size=0.2, stratify=over_y)

    classes = np.unique(le.inverse_transform(np.argmax(y_train, axis=1)))

    model = Sequential([
        Embedding(5000, 128),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Dropout(0.2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(5),
        Dropout(0.2),
        Conv1D(32, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dense(13, activation='softmax'),
    ])


    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    model.summary()
    print(y_train)
    history = model.fit(x_train, y_train,
                        epochs=100,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        batch_size=10)

    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)
    plt.clf()
    y_pred = model.predict(x_test)

    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), normalize='true')
    print(cm)
    plot_confusion_matrix(cm=cm, classes=classes, title=f"CD1 MLP simple network")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=classes))


if __name__ == '__main__':
    train_cd1_mlp(weighted=True)