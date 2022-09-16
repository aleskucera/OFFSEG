import os
import matplotlib.pyplot as plt


def plot_loss(history: dict, save_path: str) -> None:
    plt.figure()
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(save_path, 'loss.png'))


def plot_score(history: dict, save_path: str) -> None:
    plt.figure()
    plt.plot(history['train_mIoU'], label='train_mIoU', marker='*')
    plt.plot(history['val_mIoU'], label='val_mIoU', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(save_path, 'plot.png'))


def plot_acc(history: dict, save_path: str) -> None:
    plt.figure()
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(save_path, 'acc.png'))
