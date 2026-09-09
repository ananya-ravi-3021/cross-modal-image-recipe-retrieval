"""
PLM-based text embeddings for the allergen label set.

Adapted from BorLan (text_features/text_embedding.py):
    Ma, W., Li, S., Zhang, J., Liu, C. H., Kang, J., Wang, Y., & Huang, G. (2023).
    "Borrowing Knowledge From Pre-trained Language Model: A New Data-efficient
    Visual Learning Paradigm." ICCV 2023.
    Reference implementation: https://github.com/BIT-DA/BorLan (MIT License)

BorLan in turn builds on CoOp (https://github.com/KaiyangZhou/CoOp) and
Self-Tuning (https://github.com/thuml/Self-Tuning); the prompt template
structure in imagenet_templates follows CoOp.
"""

import torch
import os
import os.path as osp
import sys
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, T5Tokenizer, T5Model, XLNetTokenizer, XLNetModel, DebertaTokenizer, DebertaModel, MT5Model, GPT2Model, GPT2Tokenizer
from imagenet_templates import FOOD_TEMPLATES, FOOD_NEW_TEMPLATES, extra_templates
from utils import RestrictionsInfo

import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory for the generated .pkl. Override if yours lives elsewhere:
#     export PKL_DIR=/path/to/gcn_pkl
PKL_DIR = os.environ.get('PKL_DIR', osp.join('model', 'gcn_pkl'))
FEATURE_PKL = osp.join(PKL_DIR, 'feature_data.pkl')

# -----------------------------------------------------------------------
# Setting up here.
choice = 'BertL' # Choose language model
classnames = RestrictionsInfo().getRestrictions() # Set classnames

object_token_avg = True # use [CLS] token or word token
use_prefix = True # add prefix "This is"
# -----------------------------------------------------------------------

model_dict = {'example':['model','tokenizer','model-type','tokenizer-type'],
              'Bert':['BertModel','BertTokenizer','prajjwal1/bert-small','prajjwal1/bert-small'],
              'BertB':['BertModel','BertTokenizer','bert-base-uncased','bert-base-uncased'],
              'BertL':['BertModel','BertTokenizer','bert-large-uncased','bert-large-uncased'],
              'Deberta':['DebertaModel','DebertaTokenizer','microsoft/deberta-large','microsoft/deberta-large'],
              'XLNet':['XLNetModel','XLNetTokenizer','xlnet-large-cased','xlnet-large-cased'],
              'T5':['T5Model','T5Tokenizer','t5-large','t5-large'],
              'mT5':['MT5Model','T5Tokenizer','google/mt5-large','google/mt5-large'],
              'T5-3b':['T5Model','T5Tokenizer','t5-3b','t5-3b'],
              'gpt2':['GPT2Model','GPT2Tokenizer','gpt2','gpt2'],} #GPT suggest using text_embedding_gpt.py instead


tokenizer = eval(model_dict[choice][1]).from_pretrained(model_dict[choice][3])
model = eval(model_dict[choice][0]).from_pretrained(model_dict[choice][2]).to(device)
model.eval()

templates = FOOD_NEW_TEMPLATES

def article(name):
  return 'an' if name[0] in 'aeiouAEIOU' else 'a'

def get_text_features():
    with torch.no_grad():
        all_text_features = []
        for i, temp in enumerate(templates):
            if use_prefix:
                temp = 'This is ' + temp if temp.startswith('a') or temp.startswith('the') else temp 
            for c in classnames:
                c = c.replace('_', ' ')
                ingredient_features = []
                for x in classnames[c]:
                    x = x.replace('_', ' ')
                    prompt = temp.format(x, i)
                    tokens = tokenizer(prompt, return_tensors='pt', padding=True)
                    # print(tokens['input_ids'])
                    batch = {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}
                    # print(batch)
                    # print(tokenizer.decode(tokens['input_ids'][0]))
                    if choice in ['Bert','BertB','BertL','Deberta','XLNet','gpt2']:
                        output = model(input_ids=tokens['input_ids'].to(device), attention_mask=tokens['attention_mask'].to(device), return_dict = True)
                    else:
                        output = model.encoder(**batch, return_dict = True)
                    last_hidden_states = output.last_hidden_state
                    # print(last_hidden_states.size())
                    if object_token_avg:
                        text_features = last_hidden_states.mean(dim=1).squeeze()
                    else:
                        print('use cls token')
                        text_features = last_hidden_states[:,0,:].squeeze()
                    ingredient_features.append(text_features)
                    # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    # print(text_features.size())
                    # print("debug size", text_features.size())
                aggregated_features = torch.stack(ingredient_features).mean(dim=0)
                all_text_features.append(aggregated_features)
            # break
        all_text_features = torch.stack(tuple(all_text_features),dim=0)
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
        # feature matrix dimension: 10 * 80 * 1024
        aggregated_features, _ = torch.max(all_text_features, dim=-1)
        aggregated_features = aggregated_features.reshape((10, 80))
        return aggregated_features
    
features = get_text_features()

# Serialize the return information into a .pkl file
with open(FEATURE_PKL, 'wb') as file:
    pickle.dump(features, file)