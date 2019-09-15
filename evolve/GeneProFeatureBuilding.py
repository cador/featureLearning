# -*- coding: utf-8 -*-
"""
封装特征学习的类

@author: youhaolin
@email: cador.ai@aliyun.com
"""

from evolve.Adjust import *
from evolve.Actions import *
from feature_generator.FeaturePlot import *
import matplotlib
import sys
import copy
matplotlib.rcParams['font.family'] = 'SimHei'


class FeatureEvolution:
    def __init__(self, x, y, task='class', pop_size=100, need_genes=3, max_epochs=100, alpha=1.1,
                 cross_prob=0.85, mutate_prob=0.1, print_batch=10, top_ratio=0.4):
        self.x = x
        self.y = y
        self.task = task
        self.pop_size = pop_size
        self.need_genes = need_genes
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.print_batch = print_batch
        self.top_ratio = top_ratio

        if task not in ['class', 'reg']:
            print("task 参数错误，需要设置为 class 或 reg")
            sys.exit(0)

        if task == 'reg':
            self.std_error = evaluation_regression(self.x, self.y)
            self.handle = evaluation_regression
        else:
            self.std_error = evaluation_classify(self.x, self.y)
            self.handle = evaluation_classify

        # 产生初始种群
        self.feature_index = list(range(1, self.x.shape[1]+1))
        self.individuals = gen_individuals(self.pop_size, self.need_genes, self.x, self.feature_index)
        self.adjusts = [get_adjust(self.std_error, y, df, self.handle) for df in self.individuals['df']]

    def evolve(self):
        for k in range(self.max_epochs):
            # 0.备份父代个体
            pre_individuals = copy.deepcopy(self.individuals)
            pre_adjusts = self.adjusts.copy()

            # 1.交叉
            inter_cross(self.individuals['df'], self.individuals['gene'], self.cross_prob)

            # 2.变异
            mutate(self.individuals['df'], self.individuals['gene'], self.mutate_prob, self.x, self.feature_index)

            # 3.计算适应度
            adjusts = []
            for df in self.individuals['df']:
                adjusts.append(get_adjust(self.std_error, self.y, df, evaluation_regression))

            # 4.合并，并按adjusts降序排列，取前0.4*popSize个个体进行返回，对剩余的个体随机选取0.6*popSize个返回
            pre_gene_keys = [''.join(e) for e in pre_individuals['gene']]
            gene_keys = [''.join(e) for e in self.individuals['gene']]
            for i in range(len(pre_gene_keys)):
                key = pre_gene_keys[i]
                if key not in gene_keys:
                    self.individuals['df'].append(pre_individuals['df'][i])
                    self.individuals['gene'].append(pre_individuals['gene'][i])
                    adjusts.append(pre_adjusts[i])

            split_val = pd.Series(adjusts).quantile(q=1-self.top_ratio)
            index = list(range(len(adjusts)))
            need_delete_count = len(adjusts) - self.pop_size
            random.shuffle(index)
            indices = []
            for i in index:
                if need_delete_count > 0:
                    if adjusts[i] <= split_val:
                        indices.append(i)
                        need_delete_count = need_delete_count - 1
                else:
                    break

            self.individuals['df'] = [i for j, i in enumerate(self.individuals['df']) if j not in indices]
            self.individuals['gene'] = [i for j, i in enumerate(self.individuals['gene']) if j not in indices]
            self.adjusts = [i for j, i in enumerate(adjusts) if j not in indices]
            if np.mean(adjusts) > 0:
                alpha = np.max(adjusts) / np.mean(adjusts)
            else:
                alpha = np.max(adjusts)
            if k % self.print_batch == self.print_batch - 1 or k == 0:
                print("第 ", k + 1, " 次迭代，最大适应度为 ", np.max(adjusts), " alpha : ", alpha)
            if np.mean(adjusts) > 0 and alpha < self.alpha:
                print("第 ", k + 1, " 次迭代，最大适应度为 ", np.max(adjusts), " alpha : ", alpha)
                print("进化终止，算法已收敛！ 共进化 ", k, " 代！")
                break

    def get_feature(self):
        loc = np.argmax(self.adjusts)
        return {'df': self.individuals['df'][loc], 'gene': self.individuals['gene'][loc]}

    def plot_feature(self, fig_size=(10, 20), node_size=1000, font_size=13):
        counter = 1
        loc = np.argmax(self.adjusts)
        titles = ['特征-g'+str(i) for i in list(range(1, self.need_genes+1))]
        plt.figure(figsize=fig_size)
        for e in self.individuals['gene'][loc]:
            plt.subplot(self.need_genes, 1, counter)
            plot_tree(e, title=titles[counter - 1], node_size=node_size, font_size=font_size)
            counter = counter + 1
        plt.show()
