import numpy as np
import matplotlib.pyplot as plt
import sys

def error_from_acc(stream):
    stream['value'] = 1.0 - stream['value']


def smooth(stream, rate):
    for idx, data in enumerate(stream['value']):
        last = stream['value'][max(0, idx - 1)]
        stream['value'][idx] = last * rate + (1 - rate) * data

def load_tf_csv(name):
    return np.genfromtxt(name, delimiter=',',
                            skip_header=1,
                            skip_footer=10, names=['time', 'step', 'value'])

def show_whitening():
    neither = load_tf_csv('whitening/neither.csv')
    whiten = load_tf_csv('whitening/whiten.csv')
    norm = load_tf_csv('whitening/norm.csv')
    both = load_tf_csv('whitening/both.csv')

    streams = [neither, whiten, norm, both]
    map(lambda x: error_from_acc(x), streams)
    map(lambda x: smooth(x, 0.5), streams)
    plt.plot(neither['step'], neither['value'], '-')
    plt.plot(whiten['step'], whiten['value'], '--')
    plt.plot(norm['step'], norm['value'], '-.')
    plt.plot(both['step'], both['value'], ':')
    plt.xlim(0, 25)
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.legend(['None', 'Whiten', 'Normalize', 'Norm+Whiten'], loc='upper right')
    plt.show()
    return

def show_normalisation():
    nonorm_acc = load_tf_csv('norm/nonorm_acc.csv')
    norm_acc = load_tf_csv('norm/norm_acc.csv')
    nonorm_loss = load_tf_csv('norm/nonorm_loss.csv')
    norm_loss = load_tf_csv('norm/norm_loss.csv')

    streams = [nonorm_acc, norm_acc, nonorm_loss, norm_loss]
    map(lambda x: error_from_acc(x), streams[:2])
    map(lambda x: smooth(x, 0.5), streams)
    plt.figure(1)
    plt.subplot(221)
    plt.plot(nonorm_acc['step'], nonorm_acc['value'], '-')
    plt.plot(norm_acc['step'], norm_acc['value'], '--')
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.legend(['None', 'Batch Normalization'], loc='upper right')
    plt.subplot(222)

    plt.plot(nonorm_loss['step'], nonorm_loss['value'], '-')
    plt.plot(norm_loss['step'], norm_loss['value'], '--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['None', 'Batch Normalization'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    if sys.argv[1] == "whitening":
        show_whitening()
    else:
        show_normalisation()

