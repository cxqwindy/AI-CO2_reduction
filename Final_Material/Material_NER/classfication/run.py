import os
import re
import json
import sys
import importlib
importlib.reload(sys)
import torch
import numpy as np

from sklearn import metrics


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    method_pred = []
    material_pred = []
    product_pred = []
    product_true = []
    material_true = []
    method_true = []
    with open('./datasets/product.txt') as product:
        dict = np.load('./datasets/product_dic.npy', allow_pickle=True).item()
        product_data = product.read()
        pattern_sentence = r'\n'
        product_data = re.split(pattern_sentence,product_data)
        for i in product_data:
            data = i.split('&&')
            if len(data) == 3:
                product_true.append(data[1])
                try:
                    product_pred.append(dict[data[0]])
                except KeyError:
                    product_pred.append(dict['HCOOH'])
        print(metrics.classification_report(product_true, product_pred, zero_division = True))
    with open('./datasets/material.txt') as material:
        dict = np.load('./datasets/material_dic.npy', allow_pickle=True).item()
        material_data = material.read()
        pattern_sentence = r'\n'
        material_data = re.split(pattern_sentence,material_data)
        for i in material_data:
            data = i.split('&&')
            if len(data) == 3:
                material_true.append(data[1])
                try:
                    material_pred.append(dict[data[0]])
                except KeyError:
                    material_pred.append(dict['CuRu'])
        print(metrics.classification_report(material_true, material_pred, zero_division = True))
    with open('./datasets/method.txt') as method:
        dict = np.load('./datasets/method_dic.npy', allow_pickle=True).item()
        method_data = method.read()
        pattern_sentence = r'\n'
        method_data = re.split(pattern_sentence,method_data)
        for i in method_data:
            data = i.split('&&')
            if len(data) == 3:
                method_true.append(data[1])
                try:
                    method_pred.append(dict[data[0]])
                except KeyError:
                    method_pred.append(dict['Bi-MWCNT-COOH composite on Cu electrode'])
        print(metrics.classification_report(method_true, method_pred, zero_division = True))

def predict(input_file):
    pred = []
    with open(input_file) as input:
        input_data = input.read()
        pattern_sentence = r'\n'
        input_data = re.split(pattern_sentence,input_data)
        for i in input_data:
            data = i.split()
            if len(data) == 4:
                if data[3] == 'material':
                    dict = np.load('classfication/datasets/material_dic.npy', allow_pickle=True).item()
                    try:
                        data[2]=(dict[data[0]])
                    except KeyError:
                        data[2]=(dict['CuRu']) 
                    pred.append([data[0],data[1],data[2],data[3]])                 
                elif data[3] == 'method':
                    dict = np.load('classfication/datasets/method_dic.npy', allow_pickle=True).item()
                    try:
                        data[2]=(dict[data[0]])
                    except KeyError:
                        data[2]=(dict['Bi-MWCNT-COOH composite on Cu electrode'])
                    pred.append([data[0],data[1],data[2],data[3]]) 
                elif data[3] == 'product':
                    dict = np.load('classfication/datasets/product_dic.npy', allow_pickle=True).item()
                    try:
                        data[2]=(dict[data[0]])
                    except KeyError:
                        data[2]=(dict['HCOOH']) 
                    pred.append([data[0],data[1],data[2],data[3]]) 
                else:
                    data[2]='Faradaicefficiency'
                    pred.append([data[0],data[1],data[2],data[3]])
        return pred

