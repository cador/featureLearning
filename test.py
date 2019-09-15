# -*- coding: utf-8 -*-
"""
测试其它模块功能

@author: youhaolin
@email: cador.ai@aliyun.com
"""

import pandas as pd
import matplotlib.pyplot as plt
from feature_generator import FeaturePlot as Fp, RandomFeature as Rf
from evolve import Adjust as Aj, Actions as At
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
iris = pd.read_csv("http://image.cador.cn/data/iris.csv")
out = Rf.random_get_tree(iris, [1, 2, 3, 4])
exp_tmp = out['tree_exp']
print(exp_tmp)
print(out['f_value'])
plt.figure(figsize=(20, 11))
Fp.plot_tree(exp_tmp, node_size=1000, font_size=13)
plt.show()

gen_out = At.gen_individuals(5, 4, iris, [1, 2, 3, 4])
for x in gen_out['df']:
    print("____________________________________________")
    print(x.head(2))

for x in gen_out['gene']:
    print("____________________________________________")
    print(x[0])

X, y = iris.drop(columns='Species'), iris.Species
print(Aj.evaluation_classify(X, y))

X, y = iris.drop(columns=['Petal.Width', 'Species']), iris['Petal.Width']
print(Aj.evaluation_regression(X, y))

At.inter_cross(None, gen_out['gene'], 1)

A = ['g(add,X1,X2)', 'g(log,X1)', 'g(add,g(log,X2),X3)']
B = ['g(pow2,X3)', 'g(add,g(inv,X1),g(log,X2))', 'g(log,g(tanh,X4))']
counter = 1
titles = ['个体A基因1', '个体A基因2', '个体A基因3', '个体B基因1', '个体B基因2', '个体B基因3']
plt.figure(figsize=(15,8))
for e in A+B:
    plt.subplot(2, 3, counter)
    Fp.plot_tree(e, title=titles[counter - 1], node_size=1000, font_size=13)
    counter = counter + 1
plt.show()

At.inter_cross(None, [A,B], 1)
counter = 1
titles = ['个体A基因1', '个体A基因2', '个体A基因3', '个体B基因1', '个体B基因2', '个体B基因3']
plt.figure(figsize=(15, 8))
for e in A+B:
    plt.subplot(2, 3, counter)
    Fp.plot_tree(e, title=titles[counter - 1], node_size=1000, font_size=13)
    counter = counter + 1
plt.show()


pre_A = A.copy()
At.mutate(None, [A], 0.9, iris, [1, 2, 3, 4])
counter = 1
titles = ['个体A基因1（变异前）', '个体A基因2（变异前）', '个体A基因3（变异前）', '个体A基因1（变异后）',
          '个体A基因2（变异后）', '个体A基因3（变异后）']
plt.figure(figsize=(15, 8))
for e in pre_A+A:
    plt.subplot(2, 3, counter)
    Fp.plot_tree(e, title=titles[counter - 1], node_size=500, font_size=10)
    counter = counter + 1
plt.show()

