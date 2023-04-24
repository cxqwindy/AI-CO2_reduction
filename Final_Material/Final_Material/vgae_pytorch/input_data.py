'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData, Data
from transformers import AutoModel,AutoTokenizer
from scipy.sparse import csc_matrix,lil_matrix

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_bak(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def load_data():
    """
    从数据中
    :param
    :return:
    """
    raw_data = np.load('../data/raw_data_all.npy').tolist()
    value_data = [list(eval(i).values()) for i in raw_data]
    graph_node = []

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    for one in value_data:
        graph_node += one[:6]
    graph_node = np.array(graph_node)
    graph_node_mapping = {index_id: int(i) for i, index_id in enumerate(np.unique(graph_node))}
    adj = np.zeros([len(graph_node_mapping), len(graph_node_mapping)])
    # node_x = np.zeros([len(graph_node_mapping), 768])
    node_x = np.zeros([len(graph_node_mapping), 5])
    edge_index = []
    for one_data in raw_data:
        one_data = eval(one_data)
        material = one_data['material']
        node_x[graph_node_mapping[material], 0] = 1
        material_type = one_data['material_type']
        node_x[graph_node_mapping[material_type], 1] = 1
        product = one_data['product']
        node_x[graph_node_mapping[product], 2] = 1
        method = one_data['method']
        node_x[graph_node_mapping[method], 3] = 1
        method_type = one_data['method_type']
        node_x[graph_node_mapping[method_type], 4] = 1
        adj[graph_node_mapping[method], graph_node_mapping[material]] = 1
        adj[graph_node_mapping[material], graph_node_mapping[material_type]] = 1
        adj[graph_node_mapping[method], graph_node_mapping[product]] = 1
        adj[graph_node_mapping[method], graph_node_mapping[method_type]] = 1
    adj = csc_matrix(adj)
    features = lil_matrix(node_x)


    return adj, features



if __name__ == '__main__':
    build_own_data()
