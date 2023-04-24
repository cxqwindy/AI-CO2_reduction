import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel
from transformers import BertModel as tBert
from transformers import AutoModel
import torch
import config
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

hidden_size = 768


class model_1(nn.Module):
    """
    using only BERT to extract the word embedding from material, product and method to predict the Faradaic efficiency

    """
    def __init__(self, type_number):
        super(model_1, self).__init__()
        self.word_embedding_layer = AutoModel.from_pretrained(config.model_name)
        self.linear1 = nn.Linear(hidden_size, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()


    def _process_data(self, data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)
        output = model(data, attention_mask=mask)[1].reshape(-1, 768)
        return output

    def forward(self, material, product, method, method_type, product_type, material_type, graph_embed):
        input_word = torch.cat([material, product, method], 1)
        word_embedding = self.linear1(self._process_data(input_word, self.word_embedding_layer))
        out = self.relu(word_embedding)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.sigmoid(self.linear3(out))
        return out


class model_2(nn.Module):
    """
    using BERT embedding and type embedding
    """
    def __init__(self, type_number):
        super(model_2, self).__init__()
        self.word_embedding_layer = AutoModel.from_pretrained(config.model_name)
        self.linear1 = nn.Linear(hidden_size, 400)
        self.linear_type = nn.Linear(int(type_number), 10)
        self.linear2 = nn.Linear(410, 200)
        self.linear3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()


    def _process_data(self, data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)
        output = model(data, attention_mask=mask)[1].reshape(-1, 768)
        return output

    def forward(self, material, product, method, method_type, product_type, material_type, graph_embed):
        input_word = torch.cat([material, product, method], 1)
        word_embedding = self.linear1(self._process_data(input_word, self.word_embedding_layer))
        type_embedding = self.linear_type(torch.cat([method_type, product_type, material_type], 1))
        total = torch.cat([word_embedding, type_embedding], 1)
        out = self.relu(total)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.sigmoid(self.linear3(out))
        return out


class model_3(nn.Module):
    """
    using BERT embedding and graph embedding
    """
    def __init__(self, type_number):
        super(model_3, self).__init__()
        self.word_embedding_layer = AutoModel.from_pretrained(config.model_name)
        self.linear1 = nn.Linear(hidden_size, 400)
        self.linear_graph = nn.Linear(96, 20)
        self.linear2 = nn.Linear(420, 200)
        self.linear3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()


    def _process_data(self, data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)
        output = model(data, attention_mask=mask)[1].reshape(-1, 768)
        return output

    def forward(self, material, product, method, method_type, product_type, material_type, graph_embed):
        input_word = torch.cat([material, product, method], 1)
        word_embedding = self.linear1(self._process_data(input_word, self.word_embedding_layer))
        graph_embedding = self.linear_graph(graph_embed)
        total = torch.cat([word_embedding, graph_embedding], 1)
        out = self.relu(total)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.sigmoid(self.linear3(out))
        return out

class model_4(nn.Module):
    """
    using All embedding method
    """
    def __init__(self, type_number):
        super(model_4, self).__init__()
        self.word_embedding_layer = AutoModel.from_pretrained(config.model_name)
        self.linear1 = nn.Linear(hidden_size, 400)
        self.linear_type = nn.Linear(int(type_number), 10)
        self.linear_graph = nn.Linear(96, 20)
        self.linear2 = nn.Linear(430, 200)
        self.linear3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()


    def _process_data(self, data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)
        output = model(data, attention_mask=mask)[1].reshape(-1, 768)
        return output

    def forward(self, material, product, method, method_type, product_type, material_type, graph_embed):
        input_word = torch.cat([material, product, method], 1)
        word_embedding = self.linear1(self._process_data(input_word, self.word_embedding_layer))
        type_embedding = self.linear_type(torch.cat([method_type, product_type, material_type], 1))
        graph_embedding = self.linear_graph(graph_embed)
        total = torch.cat([word_embedding, type_embedding, graph_embedding], 1)
        out = self.relu(total)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.sigmoid(self.linear3(out))
        return out

class simple_model_tr(nn.Module):
    def __init__(self, type_number):
        super(simple_model_tr, self).__init__()
        self.l1 = nn.Linear(type_number, 200)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x


class RDrop(nn.Module):
    def __init__(self):
        super(RDrop, self).__init__()
        self.ce = nn.MSELoss()
        self.kld = nn.KLDivLoss(reduction='none')
    def forward(self, logits1, logits2, target, kl_weight=1.0):
        ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + kl_weight * kl_loss
        return loss



