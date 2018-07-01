# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


def f(x, y):
    return x**2 / 20.0 + y**2

#f(x,y)をx,yでそれぞれ偏微分した時の値
def df(x, y):
    return x / 10.0, 2.0*y


init_pos = (-7.0, 2.0) #f(x,y) =  1/2.0 * x**2 + y**2とかしたときに最初に入れる座標
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1 #subplotでグラフを別々に描画するためのインデックス 1番目のグラフ

for key in optimizers:
    optimizer = optimizers[key]
    x_history = [] #更新するたびxの値を格納するための配列
    y_history = [] #更新するたびyの値を格納するための配列
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads) #gradど学習率を使ってx,yを更新
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0 #Z > 7となる等高線（点）に対しては等高線を引かない
    #→これはZ>7となる等高線を引こうとするとf(x,y)上に点がないため等高線が一部途切れるから
    #等高線と等高線との幅はデフォルトで勝手に決まっていると思われる
    
    # plot 
    plt.subplot(2, 2, idx) #2行2列分のグラフ（＝4個）を描画する、idxで何番目のグラフか
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red") #'o-'で点線グラフ
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+') #中心点
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()