# coding: utf-8
import os, sys
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

#mnist手書きデータセットのテストデータを返す（今回は学習は行わないのでテストデータのみ）
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return x_test, t_test

#用意しておいた学習済みのネットワーク（重み、バイアスも）を返す
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

#テストデータから正解の数字（０～９）を予測
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cent = 0
    for i in range(0, len(x), batch_size):
        #100枚ずつバッチ処理
        x_batch = x[i:i+batch_size]
        y_batch = predict(network,x_batch)
        p = np.argmax(y_batch, axis=1) #最も可能性の高い要素のインデックスを取得
        accuracy_cent += np.sum(p == t[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cent) / len(x)))


