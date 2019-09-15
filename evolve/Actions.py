# -*- coding: utf-8 -*-
"""
定义遗传行为

@author: youhaolin
@email: cador.ai@aliyun.com
"""

import random
import numpy as np
from feature_generator import RandomFeature as Rf
import pandas as pd


def gen_individuals(k, gen_num, input_data, featureIdx, nMax=10):
    """产生k个个体, gen_num表示每个体对应的固定基因数量"""
    indiv_list = []
    gene_list = []
    for e in range(k):
        indiv = {}
        gene = []
        for i in range(gen_num):
            out = Rf.random_get_tree(input_data, featureIdx, nMax)
            indiv["g"+str(i+1)]=out['f_value']
            gene.append(out['tree_exp'])
        indiv = pd.DataFrame(indiv)
        indiv_list.append(indiv)
        gene_list.append(gene)
    return {"df": indiv_list, "gene": gene_list}


def inter_cross(indiv_list, gene_list, prob):
    """ 对染色体进行交叉操作 """
    gene_num = len(gene_list[0])
    ready_index = list(range(len(gene_list)))
    while len(ready_index) >= 2:
        d1 = random.choice(ready_index)
        ready_index.remove(d1)
        d2 = random.choice(ready_index)
        ready_index.remove(d2)
        if np.random.uniform(0, 1) <= prob:
            loc = random.choice(range(gene_num))
#             print(d1,d2,"exchange loc --> ",loc)
            # 对数据做交叉操作
            if indiv_list is not None:
                tmp = indiv_list[d1].iloc[:, loc]
                indiv_list[d1].iloc[:, loc] = indiv_list[d2].iloc[:, loc]
                indiv_list[d2].iloc[:, loc] = tmp
            # 对基因型做交叉操作
            tmp = gene_list[d1][loc]
            gene_list[d1][loc] = gene_list[d2][loc]
            gene_list[d2][loc] = tmp


def mutate(indiv_list, gene_list, prob, input_data, featureIdx, nMax=10):
    gene_num = len(gene_list[0])
    ready_index = list(range(len(gene_list)))
    for i in ready_index:
        if np.random.uniform(0, 1) <= prob:
            loc = random.choice(range(gene_num))
#             print(i,"mutate on --> ",loc)
            tmp = Rf.random_get_tree(input_data, featureIdx, nMax)
            if indiv_list is not None:
                indiv_list[i].iloc[:,loc] = tmp['f_value']
            gene_list[i][loc] = tmp['tree_exp']
