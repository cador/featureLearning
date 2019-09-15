# -*- coding: utf-8 -*-
"""
构建特征表达式产生器

@author: youhaolin
@email: cador.ai@aliyun.com
"""

import random
import pandas as pd
from feature_generator.Operator import *


# 定义二元运算函数的集合
two_group = ['add', 'sub', 'times', 'div']

# 定义一元运算函数的集合
one_group = ['log', 'sqrt', 'pow2', 'pow3', 'inv', 'sigmoid', 'tanh', 'relu', 'binary']


# 随机增加一元运算符
def add_one_group(feature_string, prob=0.3):
    return 'g('+random.choice(one_group)+','+feature_string+')' if np.random.uniform(0, 1) < prob else feature_string


# 构建满二叉树，并生成数学表达式
def gen_full_tree_exp(var_flag_array):
    half_n = len(var_flag_array)//2
    middle_array = []
    for i in range(half_n):
        if var_flag_array[i] == '0' and var_flag_array[i+half_n] != '0':
            middle_array.append('g('+random.choice(one_group)+','+add_one_group(var_flag_array[i+half_n])+')')
        elif var_flag_array[i] != '0' and var_flag_array[i+half_n] == '0':
            middle_array.append('g('+random.choice(one_group)+','+add_one_group(var_flag_array[i])+')')
        elif var_flag_array[i] != '0' and var_flag_array[i+half_n] != '0':
            middle_array.append('g('+random.choice(two_group)+','+add_one_group(var_flag_array[i])+','+
                                add_one_group(var_flag_array[i+half_n])+')')
    if len(middle_array) == 1:
        return add_one_group(middle_array[0])
    else:
        return gen_full_tree_exp(middle_array)


# 构建偏二叉树，并生成数学表达式
def gen_side_tree_exp(var_flag_array):
    if len(var_flag_array) == 1:
        return add_one_group(var_flag_array[0])
    else:
        var_flag_array[1] = 'g('+random.choice(two_group)+','+add_one_group(var_flag_array[0])+','+\
                            add_one_group(var_flag_array[1])+')'
        del var_flag_array[0]
        return gen_side_tree_exp(var_flag_array)


def random_get_tree(input_data, featureIdx, nMax=10):
    """
    从原始数据特征中，随机获取特征表达树
    featureIdx: 原始特征的下标数值，最小从1开始
    nMax:一次最多从特征中可放回抽样次数，默认为10
    """
    data = pd.DataFrame({"X" + str(e): input_data.iloc[:, (e - 1)].values for e in featureIdx})

    # 随机抽取N个特征下标
    N = random.choice(range(2, nMax + 1))

    # 随机决定是使用满二叉树还是偏二叉树
    if random.choice([0, 1]) == 1:
        # 选择满二叉树
        select_feature_index = [random.choice(featureIdx) for i in range(N)] + [0] * int(2 ** np.ceil(np.log2(N)) - N)
        random.shuffle(select_feature_index)
        select_feature_index = ['data.X' + str(e) + ".values" if e > 0 else '0' for e in select_feature_index]
        tree_exp = gen_full_tree_exp(select_feature_index)
    else:
        # 选择偏二叉树
        select_feature_index = ['data.X' + str(e) + ".values" for e in [random.choice(featureIdx) for i in range(N)]]
        tree_exp = gen_side_tree_exp(select_feature_index)
    return {"f_value": eval(tree_exp), "tree_exp": tree_exp.replace("data.", "").replace(".values", "")}
