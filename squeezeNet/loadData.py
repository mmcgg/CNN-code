import numpy as np
import os

def load_CIFAR_batch(filename):
    """ 载入cifar数据集的一个batch """
    import pickle
    with open(filename, 'rb') as fo:
        datadict = pickle.load(fo, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ 载入cifar全部数据 """
    xs = []
    ys = []
    for b in range(1, 6):
        f = ROOT + '/data_batch_%d' % (b,)
        X, Y = load_CIFAR_batch(f)
        xs.append(X)         #将所有batch整合起来
        ys.append(Y)
    Xtr = np.concatenate(xs) #使变成行向量,最终Xtr的尺寸为(50000,32,32,3)
    Ytr = np.concatenate(ys)
    for i in range(len(Xtr)):
        Xtr[i] = (Xtr[i] - np.mean(Xtr[i])) / np.max(Xtr[i])
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    for i in range(len(Xte)):
        Xte[i] = (Xte[i] - np.mean(Xte[i])) / np.max(Xte[i])
    return Xtr, Ytr, Xte, Yte

if __name__ == '__main__':
    cifar10_dir = 'C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    print(X_train[1])