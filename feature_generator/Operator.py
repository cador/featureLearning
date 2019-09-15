# -*- coding: utf-8 -*-
"""
定义遗传编程过程中使用到的各种算子

@author: youhaolin
@email: cador.ai@aliyun.com
"""

import numpy as np
min_number = 0.01


def g(f, a, b=None):
    """
    f: 一元或二元运算函数
    a: 第一个参数
    b: 如果f是一元运算函数，则b为空，否则代表二元运算的第二个参数
    """
    if b is None:
        return f(a)
    else:
        return f(a, b)


# 一元运算
def log(x):
    return np.sign(x)*np.log2(np.abs(x)+1)


def sqrt(x):
    return np.sqrt(x-np.min(x)+min_number)


def pow2(x):
    return x**2


def pow3(x):
    return x**3


def inv(x):
    return 1*np.sign(x)/(np.abs(x)+min_number)


def sigmoid(x):
    if np.std(x) < min_number:
        return x
    x = (x - np.mean(x))/np.std(x)
    return (1 + np.exp(-x))**(-1)


def tanh(x):
    if np.std(x) < min_number:
        return x
    x = (x - np.mean(x))/np.std(x)
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def relu(x):
    if np.std(x) < min_number:
        return x
    x = (x - np.mean(x))/np.std(x)
    return np.array([e if e > 0 else 0 for e in x])


def binary(x):
    if np.std(x) < min_number:
        return x
    x = (x - np.mean(x))/np.std(x)
    return np.array([1 if e > 0 else 0 for e in x])


# 二元运算
def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def times(x, y):
    return x * y


def div(x, y):
    return x*np.sign(y)/(np.abs(y)+min_number)
