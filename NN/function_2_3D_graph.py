'''
x[0][0]**2 + x[1][0]^2, x[0][1]**2 + x[1][1]^2,...

def function2_alt(x):
   #xは２次元を想定
   #(flatten化した)X, Yの点が並んでいるので, 2×N次元になるはず
   #変数が増えれば２以外も考えられる(y = x0^2 + x1^2 + x2^2など)
   y = np.zeros_like(x[0])
   for idx in range(x[0].size):
       y[idx] = np.sum(x[:,idx]**2) #列を抽出して列の2乗和を取る
   return y
・□.sizeは多次元配列のとき、flatten化したときの全要素数を指す
→rangeで回すとき多次元配列のときは注意
・y = np.zeros_like(x[0])で返す要素を用意できるが、
np.zeros_like(x[0])だと形がxのままで合わなくなるので注意
・この場合の返り値はflatten化されているので、meshgridで出したX,Yに
合わせるにはreshapeが必要

'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def function2_alt(x):
    y = np.zeros_like(x[0])
    for idx in range(x[0].size):
        y[idx] = np.sum(x[:, idx] ** 2)
    return y


x = np.arange(-3, 3, 0.25)
y = np.arange(-3, 3, 0.25)
X, Y = np.meshgrid(x, y)

X1d = X.flatten()
Y1d = Y.flatten()
Z1d = function2_alt(np.array([X1d, Y1d]))
Z = Z1d.reshape(X.shape)  # X,Yと形が合わないためreshape

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, Z)
# 3Dグラフ描画にはmeshgridしたものそのまま入れる
# ZはmeshgridしたX,Yに合わせる必要がある

plt.show()