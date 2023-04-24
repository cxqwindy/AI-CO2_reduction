import os
from classfication.run import predict
from pywebio.input import *
from pywebio.output import *
def data_preprocessing(input_sentence):
    with open('data/new_test.txt','w') as writer:
        samples=input_sentence.split('.')
        for sample in samples:
            words=sample.replace(',', ' ').replace('(', ' ').replace(')', ' ').split(' ')
            for word in words:
                writer.write(word+' O'+'\n')
            writer.write('\n')


def print_entity(input_sentence):
    output=[]
    output2=[]
    #data_preprocessing(input_sentence)
    #os.system('python run_ner.py --data_dir=data --bert_model=bert-large-cased --task_name=ner --output_dir=out_test_method --model_dir=out_base_method --max_seq_length=128 --do_eval --warmup_proportion=0.1')
    #os.system('python run_ner.py --data_dir=data --bert_model=bert-large-cased --task_name=ner --output_dir=out_test_material --model_dir=out_base_material --max_seq_length=128 --do_eval --warmup_proportion=0.1')
    #os.system('python run_ner.py --data_dir=data --bert_model=bert-large-cased --task_name=ner --output_dir=out_test_product --model_dir=out_base_product --max_seq_length=128 --do_eval --warmup_proportion=0.1')
    #os.system('python run_ner.py --data_dir=data --bert_model=bert-large-cased --task_name=ner --output_dir=out_test_fa --model_dir=out_base_fa --max_seq_length=128 --do_eval --warmup_proportion=0.1')
    
    with open ('out_test_method/entity_results.txt','r') as f:
        i=0
        for lines in f:
            i=i+1
            if i%2 == 0:
                if lines != '\n':
                    contents=lines.split('~')
                    if len(contents)==2:
                        output.append(contents[1].strip() + ' method')
                        output2.append(contents[1].strip()+' '+contents[0].strip() + ' method')
    with open ('out_test_material/entity_results.txt','r') as f:
        i=0
        for lines in f:
            i=i+1
            if i%2 == 0:
                if lines != '\n':
                    contents=lines.split('~')
                    if len(contents)==2:
                        output.append(contents[1].strip() + ' material')
                        output2.append(contents[1].strip()+' '+contents[0].strip() + ' material')
    with open ('out_test_product/entity_results.txt','r') as f:
        i=0
        for lines in f:
            i=i+1
            if i%2 == 0:
                if lines != '\n':
                    contents=lines.split('~')
                    if len(contents)==2:
                        output.append(contents[1].strip() +' product')
                        output2.append(contents[1].strip()+' '+contents[0].strip() + ' product')
    with open ('out_test_fa/entity_results.txt','r') as f:
        i=0
        for lines in f:
            i=i+1
            if i%2 == 0:
                if lines != '\n':
                    contents=lines.split('~')
                    #print(lines)
                    if len(contents)==2:
                        output.append(contents[1].strip() +' faradaic effiency')
                        output2.append(contents[1].strip() +' '+ contents[0].strip() +' faradaic effiency')
    with open('classfication/input.txt','w') as writer:
        for line in output2:
            writer.write(line+'\n')
    return output


if __name__=='__main__':
    #input_text=input("请输入句子：")
    input='Copper is the only pure metal electrocatalyst capable of converting carbon dioxide to hydrocarbons at significant reaction rates.'
    #print(input_text)
    output=print_entity(input)
    #classfication(input)
    output2=predict('classfication/input.txt')
    #output_text=''
    with open ('out_entity/entity_results.txt','w') as writer:
        for line in output2:
            writer.write(line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + line[3]+'\n')
        #line_text='{'+line[0]+':'+line[1]+'; class1: '+line[2]+'; class2: '+line[3]+'}\n'
        #output_text=output_text+line_text
    #put_text(output_text)