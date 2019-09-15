# -*- coding: utf-8 -*-
"""
运行一个案例

@author: youhaolin
@email: cador.ai@aliyun.com
"""

import pandas as pd
import numpy as np
from evolve.GeneProFeatureBuilding import FeatureEvolution

# 读入基础数据
data = pd.read_csv("data/cemht.csv")
X = data.drop(columns=['No', 'Y'])
y = data.Y

# 对 X1~X4 进行标准化处理
X = X.apply(lambda e: (e - np.mean(e))/np.std(e), axis=1)
print(X.head())

# 创建 FeatureEvolution 实例

f_learn = FeatureEvolution(x=X, y=y, task='reg', need_genes=2)
f_learn.evolve()
f_learn.plot_feature()

