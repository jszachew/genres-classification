import time

from matplotlib import pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    plt.clf()
    plt.style.use('default')
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
    plt.close()


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
    plt.savefig(f"MLP_dropout_sgd_history_{now}.png")
    plt.clf()
