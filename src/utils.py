import itertools
import time

import numpy as np
from matplotlib import pyplot as plt
from datetime import date

from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    ax.set_title(title, fontsize=25)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    plt.rcParams.update({'font.size': 15})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    disp.plot(ax=ax, xticks_rotation='vertical', values_format='.2f', colorbar=False)
    plt.tight_layout()
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    # plt.show()
    title_file = title.replace(" ", "_")
    plt.savefig(f"{title_file}_{now}.png")
