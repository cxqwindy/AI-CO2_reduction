from transformers import AutoTokenizer


checkpoint_token = 'allenai/scibert_scivocab_uncased'
# checkpoint_token = './bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint_token)
# tokenizer = BertTokenizer(vocab_file='./pretrained_models/vocab.txt')
# model_name = './bert-base-uncased'
model_name = "allenai/scibert_scivocab_uncased"
raw_data_path = './data'
str_index = ['material', 'product', 'method', 'method_type', 'product_type', 'material_type', 'label']
str_max_len = [15, 5, 30, 5, 5, 5, 5]
accumulation_steps = 4
epoch = 100