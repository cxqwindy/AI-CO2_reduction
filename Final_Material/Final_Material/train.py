"""
BERT embedding + Type embedding + Graph embedding
"""
import torch
import os
import shutil
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import Adam
from vgae_pytorch.preprocessing import *
from torch.utils.data import DataLoader,random_split,TensorDataset
import vgae_pytorch.model as model_vage
from model import *
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder
import config as config
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csc_matrix,lil_matrix
from transformers import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm, linear_model


def _preprocess_sample(sample_str, Z, node_mapping, material_type_mapping, method_type_mapping, product_type_mapping):
    """
    preprocess each sample with the limitation of maximum length and pad each sample to maximum length
    :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
    :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
    """
    raw_sample = json.loads(sample_str)
    sample = [[] for n in range(len(config.str_index)+1)]
    for k in raw_sample.keys():
        if k == 'label':
            sample[config.str_index.index(k)].append(float(raw_sample[k])/100)
        elif k in ['method_type', 'material_type', 'product_type']:
            mapping = eval(k + '_mapping')
            initial_encode = np.zeros([1, len(mapping)])
            initial_encode[0, mapping[raw_sample[k]]] = 1
            sample[config.str_index.index(k)] = initial_encode.tolist()[0]
            sample[len(config.str_index)] += Z[node_mapping[raw_sample[k]]].tolist()
        elif k in ['material', 'product', 'method']:
            sample[config.str_index.index(k)] = config.tokenizer.encode(raw_sample[k],
                                                                 max_length=config.str_max_len[config.str_index.index(k)],
                                                                 pad_to_max_length=True)
            sample[len(config.str_index)] += Z[node_mapping[raw_sample[k]]].tolist()
        else:
            sample[config.str_index.index(k)] = config.tokenizer.encode(str(raw_sample[k]),
                                                                 max_length=config.str_max_len[config.str_index.index(k)],
                                                                 pad_to_max_length=True)

    return sample


def build_graph_data_from_raw_data():
    raw_data = np.load('./data/raw_data_revised.npy').tolist()
    value_data = [list(eval(i).values()) for i in raw_data]
    graph_node = []
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
        """
        build the relationship
        """
        adj[graph_node_mapping[method], graph_node_mapping[material]] = 1
        adj[graph_node_mapping[material], graph_node_mapping[method]] = 1
        adj[graph_node_mapping[material], graph_node_mapping[material_type]] = 1
        adj[graph_node_mapping[method], graph_node_mapping[product]] = 1
        adj[graph_node_mapping[product], graph_node_mapping[method]] = 1
        adj[graph_node_mapping[method], graph_node_mapping[method_type]] = 1
    adj = csc_matrix(adj)
    features = lil_matrix(node_x)

    return adj, features, graph_node_mapping


def train_GAE(model, optimizer, train_pos_edge_index, x):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    # if args.variational:
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test_GAE(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def pre_train_graph():

    adj, features, graph_node_mapping = build_graph_data_from_raw_data()

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                         torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                        torch.FloatTensor(features[1]),
                                        torch.Size(features[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    # init model and optimizer
    model = getattr(model_vage, 'VGAE')(adj_norm)
    optimizer = Adam(model.parameters(), lr=0.01)

    def get_scores(edges_pos, edges_neg, adj_rec):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    # train model
    for epoch in range(100):
        t = time.time()

        A_pred = model(features)
        optimizer.zero_grad()
        loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / A_pred.size(0) * (
                        1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()
        loss -= kl_divergence

        loss.backward()
        optimizer.step()

        train_acc = get_acc(A_pred, adj_label)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
              "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))
    Z = model.encode(features)
    return Z, graph_node_mapping


def validate(val_loader, model, criterion, compare=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(val_loader):
            material = Variable(material).to(device)
            product = Variable(product).to(device)
            method = Variable(method).to(device)
            method_type = Variable(method_type).to(device)
            product_type = Variable(product_type).to(device)
            material_type = Variable(material_type).to(device)
            graph_embed = Variable(graph_embed).to(device)
            target = Variable(target.view(-1, 1)).to(device)
            if compare:
                data = Variable(one_hot_embedding).to(device)
                output = model(data)
            else:
                material = Variable(material).to(device)
                product = Variable(product).to(device)
                method = Variable(method).to(device)
                method_type = Variable(method_type).to(device)
                product_type = Variable(product_type).to(device)
                material_type = Variable(material_type).to(device)
                graph_embed = Variable(graph_embed).to(device)
                target = Variable(target.view(-1, 1)).to(device)
                output = model(material, product, method, method_type, product_type, material_type, graph_embed)
            loss = criterion(output, target)
            val_loss.append(loss.item())
        return np.mean(val_loss)

def load_data_from_npy(data_file_name):
    return np.load(data_file_name).tolist()

def build_data():
    graph_embedding_Z, node_mapping = pre_train_graph()
    corpus_file = load_data_from_npy(config.raw_data_path + '/raw_data_revised.npy')
    all_sample = generate_onehot(corpus_file)
    one_hot_embedding, input_labels = [i[0] for i in all_sample], [i[1] for i in all_sample]

    material_type = np.unique(np.array([eval(i)['material_type'] for i in corpus_file]))
    material_type_mapping = {index_id: int(i) + 0 for i, index_id in enumerate(np.unique(material_type))}

    method_type = np.unique(np.array([eval(i)['method_type'] for i in corpus_file]))
    method_type_mapping = {index_id: int(i) + 0 for i, index_id in enumerate(np.unique(method_type))}

    product_type = np.unique(np.array([eval(i)['product_type'] for i in corpus_file]))
    product_type_mapping = {index_id: int(i) + 0 for i, index_id in enumerate(np.unique(product_type))}

    token_file = [_preprocess_sample(i, graph_embedding_Z, node_mapping, material_type_mapping, method_type_mapping, product_type_mapping) for
                  i in corpus_file]
    material = [i[0] for i in token_file]
    product = [i[1] for i in token_file]
    method = [i[2] for i in token_file]
    method_type = [i[3] for i in token_file]
    product_type = [i[4] for i in token_file]
    material_type = [i[5] for i in token_file]
    label = [i[6] for i in token_file]
    graph_embed = [i[7] for i in token_file]
    # when build the dataset, the one_hot_embedding will be used for compared method, such as LR\SVR..etc
    all_dataset = TensorDataset(torch.LongTensor(material),
                                torch.LongTensor(product),
                                torch.LongTensor(method),
                                torch.FloatTensor(method_type),
                                torch.FloatTensor(product_type),
                                torch.FloatTensor(material_type),
                                torch.FloatTensor(graph_embed),
                                torch.FloatTensor(label),
                                torch.FloatTensor(one_hot_embedding))
    train_len = int(len(all_dataset) * 0.8)
    test_len = int(len(all_dataset) * 0.1)
    valid_len = len(all_dataset) - train_len - test_len
    train_dataset, valid_dataset, test_dataset = random_split(all_dataset, [train_len, valid_len, test_len])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=20,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=5,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=5,
                             shuffle=True)

    return train_loader, valid_loader, test_loader, len(material_type[0]) + len(method_type[0]) + len(product_type[0])


def _preprocess_sample_tradition(sample_str, str_type):
    """
    preprocess each sample with the limitation of maximum length and pad each sample to maximum length
    :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
    :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
    """
    data_type = np.array([json.loads(i)[str_type] for i in sample_str])
    data_type_encoder = LabelEncoder()
    data_type_encoded = data_type_encoder.fit_transform(data_type)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = data_type_encoded.reshape(len(data_type_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


def generate_onehot(corpus_file):
    sample = []
    method_type = _preprocess_sample_tradition(corpus_file, 'method_type')
    material_type = _preprocess_sample_tradition(corpus_file, 'material_type')
    product = _preprocess_sample_tradition(corpus_file, 'product')
    for data_id, data in enumerate(corpus_file):
        data = json.loads(data)
        label = 0
        data_one = []
        for k in data.keys():
            if k == 'label':
                label = data[k] / 100
            elif k in ['method_type', 'material_type', 'product']:
                data_one_ = eval(k + '[data_id].tolist()')
                data_one += data_one_
            else:
                pass
        sample.append([data_one, [label]])
    return sample


def train(model_type, train_loader, test_loader, valid_loader, type_number, loss, path):
    """

    :param model_type: 1: BERT 2: BERT + Type Embedding 3:BERT + Graph Embedding 4:ALL
    :return:
    """

    # train_loader, test_loader = data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simple_model_ = eval('model_' + str(model_type) + '(' + str(type_number) + ')')
    simple_model_.to(device)
    simple_model_.train()
    torch.backends.cudnn.enabled = False
    if loss == 'MSE':
        criterion = nn.MSELoss()
    if loss == 'Rdrop':
        criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    optimizer = optim.Adam(simple_model_.parameters(), lr=1e-5)

    best_valid_loss = 10000000
    early_stop_max = 7
    early_stop = 0
    Loss = []
    for i in range(config.epoch):
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(train_loader):
            material = Variable(material).to(device)
            product = Variable(product).to(device)
            method = Variable(method).to(device)
            method_type = Variable(method_type).to(device)
            product_type = Variable(product_type).to(device)
            material_type = Variable(material_type).to(device)
            graph_embed = Variable(graph_embed).to(device)
            target = Variable(target.view(-1, 1)).to(device)
            optimizer.zero_grad()
            output = simple_model_(material, product, method, method_type, product_type, material_type, graph_embed)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if ((batch_idx + 1) % config.accumulation_steps) == 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    i + 1, batch_idx, len(train_loader), 100. *
                    batch_idx / len(train_loader), loss.item()
                ))
                running_loss = loss.item()
                Loss.append(running_loss)

        val_loss = validate(valid_loader, simple_model_, criterion)
        print('val_loss:' + str(val_loss))
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(simple_model_.state_dict(), path + str(model_type) + 'model.pt')
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= early_stop_max:
            break

    del simple_model_
    torch.cuda.empty_cache()
    Loss0 = torch.tensor(Loss)
    torch.save(Loss0, path + str(model_type) + 'epoch_{}'.format(config.epoch))
    model_best = eval('model_' + str(model_type) + '(' + str(type_number) + ')')
    model_best.load_state_dict(state_dict=torch.load(path + str(model_type) + 'model.pt'))
    model_best.to(device)
    total_loss = 0
    correct = 0
    total = 0
    test_result = []
    with torch.no_grad():
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(test_loader):
            material = Variable(material).to(device)
            product = Variable(product).to(device)
            method = Variable(method).to(device)
            method_type = Variable(method_type).to(device)
            product_type = Variable(product_type).to(device)
            material_type = Variable(material_type).to(device)
            graph_embed = Variable(graph_embed).to(device)
            target = Variable(target.view(-1, 1)).to(device)
            output = model_best(material, product, method, method_type, product_type, material_type, graph_embed)
            test_result.append([output.tolist()[0], target.tolist()[0]])
            total_loss += criterion(output, target)
            total += 1
    y_test = [i[1] for i in test_result]
    y_predict = [i[0] for i in test_result]
    print(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    del model_best
    torch.cuda.empty_cache()
    return Loss0, test_result, [mse, mae, r2]

def compare_train(model_type, train_loader, test_loader, valid_loader, type_number):
    # train_loader, test_loader = data_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_result = []
    if model_type == 'MLP':
        simple_model_ = simple_model_tr(type_number)
        model_best = simple_model_tr(type_number)
        simple_model_.to(device)
        simple_model_.train()
        early_stop = 0
        criterion = nn.MSELoss()
        best_valid_loss = 100000
        optimizer = optim.Adam(simple_model_.parameters(), lr=1e-5)
        for i in range(config.epoch):
            for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(train_loader):
                data, target = Variable(one_hot_embedding).to(device), Variable(target.view(-1, 1)).to(device)
                output = simple_model_(data)
                pred = output
                loss = criterion(output, target)
                loss = loss / config.accumulation_steps
                loss.backward()
                if ((batch_idx + 1) % config.accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                if ((batch_idx + 1) % config.accumulation_steps) == 1:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                        i + 1, batch_idx, len(train_loader), 100. *
                        batch_idx / len(train_loader), loss.item()
                    ))
            val_loss = validate(valid_loader, simple_model_, criterion, compare=True)
            print('val_loss:' + str(val_loss))
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                model_best = simple_model_
                early_stop = 0
            else:
                early_stop += 1
            if early_stop >= 6:
                break

        model_best.eval()
        with torch.no_grad():
            for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(test_loader):
                data = one_hot_embedding.to(device)
                target = target.to(device)
                mask = []
                for sample in data:
                    mask.append([1 if i != 0 else 0 for i in sample])
                output = model_best(data)
                test_result.append([output.tolist()[0], target.tolist()[0]])
    if model_type == 'SVR':

        regr = svm.SVR()
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(train_loader):
            regr.fit(one_hot_embedding.tolist(), target.reshape(1, -1).tolist()[0])
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(test_loader):
            output = torch.tensor(regr.predict(one_hot_embedding.tolist()))
            test_result.append([output.tolist()[0], target.tolist()[0]])
            print(output, target)


    if model_type == 'LinearRegression':

        regr = LinearRegression()
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(train_loader):
            regr.fit(one_hot_embedding.tolist(), target.reshape(1, -1).tolist()[0])
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(test_loader):
            output = torch.tensor(regr.predict(one_hot_embedding.tolist()))
            print(output, target)
            test_result.append([output.tolist()[0], target.tolist()[0]])


    if model_type == 'Bayesian Ridge Regression':

        regr = linear_model.BayesianRidge()
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(train_loader):
            regr.fit(one_hot_embedding.tolist(), target.reshape(1, -1).tolist()[0])
        for batch_idx, (material, product, method, method_type, product_type, material_type, graph_embed, target, one_hot_embedding) in enumerate(test_loader):
            output = torch.tensor(regr.predict(one_hot_embedding.tolist()))
            print(output, target)
            test_result.append([output.tolist()[0], target.tolist()[0]])

    y_test = [i[1] for i in test_result]
    y_predict = [i[0] for i in test_result]
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    print(mse, mae, r2)
    return test_result, [mse, mae, r2]


if __name__ == '__main__':
    loss = 'MSE'
    path = 'result/'
    train_loader = torch.load('train_data/train_loader.pt')
    valid_loader = torch.load('train_data/valid_loader.pt')
    test_loader = torch.load('train_data/test_loader.pt')
    type_number = 36
    train_loss_all, test_result_all, test_loss_all = train(4, train_loader, test_loader, valid_loader, type_number,
                                                           loss, path)
    train_loss_bert_and_type, test_result_bert_and_type, test_loss_bert_and_type = train(2, train_loader,
                                                                                         test_loader,
                                                                                         valid_loader,
                                                                                         type_number, loss,
                                                                                         path)

    train_loss_bert_and_graph, test_result_bert_and_graph, test_loss_bert_and_graph = train(3, train_loader,
                                                                                            test_loader,
                                                                                            valid_loader,
                                                                                            type_number, loss,
                                                                                            path)
    train_loss_just_bert, test_result_just_bert, test_loss_just_bert = train(1, train_loader, test_loader,
                                                                             valid_loader, type_number, loss,
                                                                             path)

    test_svr, loss_svr = compare_train('SVR', train_loader, test_loader, valid_loader, type_number)
    test_lr, loss_lr = compare_train('LinearRegression', train_loader, test_loader, valid_loader, type_number)
    test_brr, loss_brr = compare_train('Bayesian Ridge Regression', train_loader, test_loader, valid_loader,
                                       type_number)
    test_mlp, loss_mlp = compare_train('MLP', train_loader, test_loader, valid_loader, type_number)
    with open(path + 'result.txt', 'w+') as f:
        f.write('just BERT\n')
        f.write('train_loss\n')
        f.write(str(train_loss_just_bert) + '\n')
        f.write('test_result\n')
        f.write(str(test_result_just_bert) + '\n')
        f.write('test_loss\n')
        f.write(str(test_loss_just_bert) + '\n')
        f.write('BERT + Type:\n')
        f.write('train_loss\n')
        f.write(str(train_loss_bert_and_type) + '\n')
        f.write('test_result\n')
        f.write(str(test_result_bert_and_type) + '\n')
        f.write('test_loss\n')
        f.write(str(test_loss_bert_and_type) + '\n')
        f.write('BERT + Graph\n')
        f.write('train_loss\n')
        f.write(str(train_loss_bert_and_graph) + '\n')
        f.write('test_result\n')
        f.write(str(test_result_bert_and_graph) + '\n')
        f.write('test_loss\n')
        f.write(str(test_loss_bert_and_graph) + '\n')
        f.write('ALL\n')
        f.write('train_loss\n')
        f.write(str(train_loss_all) + '\n')
        f.write('test_result\n')
        f.write(str(test_result_all) + '\n')
        f.write('test_loss\n')
        f.write(str(test_loss_all) + '\n')
        f.write('SVR\n')
        f.write('test_result\n')
        f.write(str(test_svr) + '\n')
        f.write('test_loss\n')
        f.write(str(loss_svr) + '\n')
        f.write('LinearRegression\n')
        f.write('test_result\n')
        f.write(str(test_lr) + '\n')
        f.write('test_loss\n')
        f.write(str(loss_lr) + '\n')
        f.write('Bayesian Ridge Regression\n')
        f.write('test_reslut\n')
        f.write(str(test_brr) + '\n')
        f.write('test_loss\n')
        f.write(str(loss_brr) + '\n')
        f.write('MLP\n')
        f.write('test_result\n')
        f.write(str(test_mlp) + '\n')
        f.write('test_loss\n')
        f.write(str(loss_mlp) + '\n')





