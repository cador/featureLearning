# featureLearning
基于遗传编程的方法实现特征学习

实现原理，请参考书籍**《Python预测之美：数据分析与算法实战》**

## 使用方法
```python
import pandas as pd
import numpy as np
from evolve.GeneProFeatureBuilding import FeatureEvolution

```

#### 读入基础数据
```python
data = pd.read_csv("data/cemht.csv")
X = data.drop(columns=['No', 'Y'])
y = data.Y
```

#### 进行标准化处理
```python
X = X.apply(lambda e: (e - np.mean(e))/np.std(e), axis=1)
print(X.head())
```


#### 创建 FeatureEvolution 实例
```python
f_learn = FeatureEvolution(x=X, y=y, task='reg', need_genes=2)

```

####  进化迭代
```python
f_learn.evolve()
```

####  获取学习到的特征数据，并绘制特征的二叉树图
```python
out = f_learn.get_feature()
f_learn.plot_feature()
```
