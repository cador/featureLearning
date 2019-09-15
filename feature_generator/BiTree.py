# -*- coding: utf-8 -*-
"""
定义二叉树节点类及相关操作

@author: youhaolin
@email: cador.ai@aliyun.com
"""

import re


class Node:
    def __init__(self, value, label, left=None, right=None):
        self.value = value
        self.label = label
        self.left = left
        self.right = right


def transform(feature_string):
    my_dict = {}
    pattern = r'g\([^\(\)]*\)'
    so = re.search(pattern, feature_string)
    while so:
        start, end = so.span()
        key = len(my_dict)
        my_dict[key] = so.group()
        feature_string = feature_string[0:start]+'<'+str(key)+'>'+feature_string[end:]
        so = re.search(pattern, feature_string)
    return my_dict


def parse(group_unit):
    tmp = group_unit.lstrip("g(").rstrip(")").split(',')
    tmp = tmp + [None] if len(tmp) == 2 else tmp
    return [int(x[1:-1]) if x is not None and re.match(r'<[0-9]+>', x) else x for x in tmp]


def tree(mapping, start_no, index=0, labels={}):
    name, left, right = parse(mapping[start_no])
    if left is not None:
        if type(left) == int:
            left_node, s_labels, max_index = tree(mapping, left, index + 1, labels)
            labels = s_labels
        else:
            left_node = Node(index + 1, left)
            labels[index + 1] = left
            max_index = index + 1
    else:
        left_node = None
        max_index = index

    if right is not None:
        if type(right) == int:
            right_node, s_labels, max_index = tree(mapping, right, max_index + 1, labels)
            labels = s_labels
        else:
            right_node = Node(max_index + 1, right)
            labels[max_index + 1] = right
            max_index = max_index + 1
    else:
        right_node = None

    labels[index] = name
    return Node(index, name, left_node, right_node), labels, max_index
