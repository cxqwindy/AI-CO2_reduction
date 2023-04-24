import os
import re
import json
import random
from gensim.models import word2vec
from gensim.models import KeyedVectors
import gensim
import logging
import numpy as np

random.seed(12345)

here = os.path.dirname(os.path.abspath(__file__))


def convert_data(line):

    head_name, relation, text = re.split(r'&&', line)
    head_name = head_name.replace('(','（').replace('[','（').replace(')','）').replace(']','）').replace('+','sp').replace('*/','sp1')
    text = text.replace('(','（').replace('[','（').replace(')','）').replace(']','）').replace('+','sp').replace('*/','sp1')

    match_obj1 = re.search(head_name, text)
    if match_obj1:  # 姑且使用第一个匹配的实体的位置
        head_pos = match_obj1.span()
        item = {
            'h': {
                'name': head_name,
                'pos': head_pos
            },
            'relation': relation,
            'text': text
        }
        return item
    else:
        return None


def save_data(lines, file):
    print('保存文件：{}'.format(file))
    with open(file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            item = convert_data(line)
            if item is None:
                continue
            json_str = json.dumps(item, ensure_ascii=False)
            f_out.write('{}\n'.format(json_str))


def split_data_material(file):
    file_dir = os.path.dirname(file)
    train_file = os.path.join(file_dir, 'train_material.jsonl')
    val_file = os.path.join(file_dir, 'val_material.jsonl')
    with open(file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    lines = [line.strip() for line in lines]
    random.shuffle(lines)
    lines_len = len(lines)
    sentences = [i.split('&&') for i in lines]
    list1 = []
    list2 = []
    for i in sentences:
        list1.append(i[0])
        list2.append(i[1])
    dic = dict(zip(list1, list2))
    np.save('material_dic.npy', dic)
    train_lines = lines[:lines_len * 7 // 10]
    val_lines = lines[lines_len * 7 // 10:]
    save_data(train_lines, train_file)
    save_data(val_lines, val_file)

def split_data_method(file):
    file_dir = os.path.dirname(file)
    train_file = os.path.join(file_dir, 'train_method.jsonl')
    val_file = os.path.join(file_dir, 'val_method.jsonl')
    with open(file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    lines = [line.strip() for line in lines]
    random.shuffle(lines)
    lines_len = len(lines)
    sentences = [i.split('&&') for i in lines]
    list1 = []
    list2 = []
    for i in sentences:
        list1.append(i[0])
        list2.append(i[1])
    dic = dict(zip(list1, list2))
    np.save('method_dic.npy', dic)
    train_lines = lines[:lines_len * 7 // 10]
    val_lines = lines[lines_len * 7 // 10:]
    save_data(train_lines, train_file)
    save_data(val_lines, val_file)

def split_data_product(file):
    file_dir = os.path.dirname(file)
    train_file = os.path.join(file_dir, 'train_product.jsonl')
    val_file = os.path.join(file_dir, 'val_product.jsonl')
    with open(file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    lines = [line.strip() for line in lines]
    random.shuffle(lines)

    lines_len = len(lines)
    train_lines = lines[:lines_len * 7 // 10]
    sentences = [i.split('&&') for i in lines]
    list1 = []
    list2 = []
    for i in sentences:
        list1.append(i[0])
        list2.append(i[1])
    dic = dict(zip(list1, list2))
    np.save('product_dic.npy', dic)
    val_lines = lines[lines_len * 7 // 10:]
    save_data(train_lines, train_file)
    save_data(val_lines, val_file)


def main():
    material_data = os.path.join(here, 'material.txt')
    split_data_material(material_data)
    method_data = os.path.join(here, 'method.txt')
    split_data_method(method_data)
    product_data = os.path.join(here, 'product.txt')
    split_data_product(product_data)


if __name__ == '__main__':
    main()
