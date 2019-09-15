# -*- coding: utf-8 -*-
"""
绘制特征表达式对应的二叉树

@author: youhaolin
@email: cador.ai@aliyun.com
"""

import networkx as nx
import matplotlib.pyplot as plt
from feature_generator import BiTree as Bt


def create_graph(G, node, pos={}, x=0, y=0, layer=1):
    pos[node.value] = (x, y)
    if node.left:
        G.add_edge(node.value, node.left.value)
        l_x, l_y = x - 1 / layer, y - 1
        l_layer = layer + 1
        create_graph(G, node.left, x=l_x, y=l_y, pos=pos, layer=l_layer)
    if node.right:
        G.add_edge(node.value, node.right.value)
        r_x, r_y = x + 1 / layer, y - 1
        r_layer = layer + 1
        create_graph(G, node.right, x=r_x, y=r_y, pos=pos, layer=r_layer)
    return G, pos


def plot_tree(feature_string, title=None, node_size=5000, font_size=18):
    my_dict = Bt.transform(feature_string)
    root, labels, _ = Bt.tree(my_dict, len(my_dict)-1, 0, labels={})
    graph = nx.Graph()
    graph, pos = create_graph(graph, root)
    nx.draw_networkx(graph, pos, node_size=node_size, width=2, node_color='black', font_color='white',
                     font_size=font_size, with_labels=True, labels=labels)
    plt.axis('off')
    if title is not None:
        plt.title(title)
