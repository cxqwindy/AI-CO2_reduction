from collections import namedtuple
from ntpath import join
import pandas as pd
import numpy as np
import os
import json
import random
import sys
import importlib
importlib.reload(sys)



def split_data(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    lines = [line.strip() for line in lines]
    random.shuffle(lines)

def flatten(l):
  return [item for sublist in l for item in sublist]

def split_document_example(example):
    split_examples = []
    sent_example = {
        "data": example["data"],
        "label": example["label"],
      }
    split_examples.append(sent_example)
    return split_examples

def read_data():
    file = open('all0702.jsonl','r',encoding='utf-8') 
    train_examples = [json.loads(jsonline) for jsonline in file.readlines()]
    doc_examples = []
    for doc_id, example in enumerate(train_examples):
        doc_examples.append([])
        for e in split_document_example(example):
            e["doc_id"] = doc_id + 1
            doc_examples[-1].append(e)
    return doc_examples
    
def main():
    input_file='label_config.xlsx'
    label_config=pd.read_excel(input_file)
    material = label_config['material'].tolist()
    method = label_config['method'].tolist()
    while np.nan in method:
        method.remove(np.nan)
    product = label_config['product'].tolist()
    while np.nan in product:
        product.remove(np.nan)
    Faradaicefficiency = label_config['Faradaicefficiency'].tolist()
    while np.nan in Faradaicefficiency:
        Faradaicefficiency.remove(np.nan)
    final_data = read_data()
    with open("./datasets/material.txt", "w+", encoding='utf-8') as f_out:
        for item in final_data:
            item = item[0]
            sen = item['data']
            if item['label']:
                current = 0
                for label_num in item['label']:
                    left_index = label_num[0]
                    right_index = label_num[1]
                    str_label = label_num[2]
                    entity = sen[left_index:right_index]
                    if str_label in material:
                        f_out.write(entity.replace('\n', ''))
                        f_out.write('&&')
                        f_out.write(str_label)
                        f_out.write('&&')
                        f_out.write(sen.replace('Abstract:', '').replace('Title:', '').replace('\n', ' '))
                        f_out.write('\n')
    with open("./datasets/method.txt", "w+", encoding='utf-8') as f_out:
        for item in final_data:
            item = item[0]
            sen = item['data']
            if item['label']:
                current = 0
                for label_num in item['label']:
                    left_index = label_num[0]
                    right_index = label_num[1]
                    str_label = label_num[2]
                    entity = sen[left_index:right_index]
                    if str_label in method:
                        f_out.write(entity.replace('\n', ''))
                        f_out.write('&&')
                        f_out.write(str_label)
                        f_out.write('&&')
                        f_out.write(sen.replace('Abstract:', '').replace('Title:', '').replace('\n', ' '))
                        f_out.write('\n')
    with open("./datasets/product.txt", "w+", encoding='utf-8') as f_out:
        for item in final_data:
            item = item[0]
            sen = item['data']
            if item['label']:
                current = 0
                for label_num in item['label']:
                    left_index = label_num[0]
                    right_index = label_num[1]
                    str_label = label_num[2]
                    entity = sen[left_index:right_index]
                    if str_label in product:
                        f_out.write(entity.replace('\n', ''))
                        f_out.write('&&')
                        f_out.write(str_label)
                        f_out.write('&&')
                        f_out.write(sen.replace('Abstract:', '').replace('Title:', '').replace('\n', ' '))
                        f_out.write('\n')
    with open("./datasets/Faradaicefficiency.txt", "w+", encoding='utf-8') as f_out:
        for item in final_data:
            item = item[0]
            sen = item['data']
            if item['label']:
                current = 0
                for label_num in item['label']:
                    left_index = label_num[0]
                    right_index = label_num[1]
                    str_label = label_num[2]
                    entity = sen[left_index:right_index]
                    if str_label in Faradaicefficiency:
                        f_out.write(entity.replace('\n', ''))
                        f_out.write('&&')
                        f_out.write(str_label)
                        f_out.write('&&')
                        f_out.write(sen.replace('Abstract:', '').replace('Title:', '').replace('\n', ' '))
                        f_out.write('\n')
    with open('./datasets/materials.txt', "w+", encoding='utf-8') as label_file:
        for label in material:
                label_file.write(label +'\n')
    with open('./datasets/methods.txt', "w+", encoding='utf-8') as label_file:
        for label in method:
                label_file.write(label +'\n')
    with open('./datasets/products.txt', "w+", encoding='utf-8') as label_file:
        for label in product:          
                label_file.write(label +'\n')
    with open('./datasets/Faradaicefficiency.txt', "w+", encoding='utf-8') as label_file:
        for label in Faradaicefficiency:
                label_file.write(label +'\n')



if __name__ == '__main__':
    main()



