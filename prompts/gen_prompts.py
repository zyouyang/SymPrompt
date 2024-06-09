from tokenizers import CharBPETokenizer
import pandas as pd
import dgl
from dgl.data import citation_graph
from dgl.nn.pytorch import SAGEConv
from dgl.sampling import random_walk
from torch.nn import Embedding
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import BertConfig, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F
import math
import pickle

path_entity_embeddings = 'my_data/entity_embeddings.pth'
path_relation_embeddings = 'my_data/relation_embeddings.pth'
path_entity2id = 'my_data/entity2id.txt'
path_relation2id = 'my_data/relation2id.txt'
path_train_path ='my_data/train_40.txt'
path_test_path = 'my_data/test_40.txt'
path_dev_path = 'my_data/dev_40.txt'

entity_embeddings = torch.load( path_entity_embeddings)['weight']
relation_embeddings = torch.load( path_relation_embeddings)['weight']

df_entity2id = pd.read_csv(path_entity2id, sep='\t' , header=None)
df_relation2id = pd.read_csv(path_relation2id, sep='\t' , header=None)

tokenizer = CharBPETokenizer(split_on_whitespace_only=True)
tokenizer.add_tokens(df_entity2id[0].tolist())
tokenizer.add_tokens(df_relation2id[0].tolist())

def get_token_id(node_id,tokenizer):
    return tokenizer.encode(str(node_id)).ids[0]

# ===============================================
# Prepare data for the graph
# ===============================================
entity_type={}
entity_type_num=0

type_entity = {}
all_set = set()
node_type ={}
rel_type = []
rel_s_type = []
rel_s_id = []
rel_t_type = []
rel_t_id = []
for i in range(tokenizer.get_vocab_size()):
    all_set.add(i)

# Train
dftrain = pd.read_csv(path_train_path, sep='\t' , header=None)
for i, row in dftrain.iterrows():
    s = tokenizer.encode(str(row.iloc[-3])).ids[0]
    t = tokenizer.encode(str(row.iloc[-2])).ids[0]
    r = tokenizer.encode(str(row.iloc[-1])).ids[0] 

    st,tt = str(row.iloc[-1]).split('_')[0],str(row.iloc[-1]).split('_')[1]
    if st not in entity_type:
        entity_type[st] = entity_type_num
        entity_type_num+=1
    if tt not in entity_type:
        entity_type[tt] = entity_type_num
        entity_type_num+=1

    node_type[row.iloc[-3]] = entity_type[st]
    node_type[row.iloc[-2]] = entity_type[tt]

    rel_type.append(r)
    rel_type.append(r+1)
    rel_s_type.append(entity_type[st])
    rel_s_id.append(row.iloc[-3])
    rel_s_type.append(entity_type[tt])
    rel_s_id.append(row.iloc[-2])
    rel_t_type.append(entity_type[tt])
    rel_t_id.append(row.iloc[-2])
    rel_t_type.append(entity_type[st])
    rel_t_id.append(row.iloc[-3])

dftrain = pd.read_csv(path_test_path, sep='\t' , header=None)
for i, row in dftrain.iterrows():
    s = tokenizer.encode(str(row.iloc[-3])).ids[0]
    t = tokenizer.encode(str(row.iloc[-2])).ids[0]
    r = tokenizer.encode(str(row.iloc[-1])).ids[0]
    st,tt = str(row.iloc[-1]).split('_')[0],str(row.iloc[-1]).split('_')[1]
    if st not in entity_type:
        entity_type[st] = entity_type_num
        entity_type_num+=1
    if tt not in entity_type:
        entity_type[tt] = entity_type_num
        entity_type_num+=1
    node_type[row.iloc[-3]] = entity_type[st]
    node_type[row.iloc[-2]] = entity_type[tt]

# Dev
dfdev = pd.read_csv(path_dev_path, sep='\t' , header=None)
for i, row in dfdev.iterrows():
    s = tokenizer.encode(str(row.iloc[-3])).ids[0]
    t = tokenizer.encode(str(row.iloc[-2])).ids[0]
    r = tokenizer.encode(str(row.iloc[-1])).ids[0]
    st,tt = str(row.iloc[-1]).split('_')[0],str(row.iloc[-1]).split('_')[1]
    if st not in entity_type:
        entity_type[st] = entity_type_num
        entity_type_num+=1
    if tt not in entity_type:
        entity_type[tt] = entity_type_num
        entity_type_num+=1
    node_type[row.iloc[-3]] = entity_type[st]
    node_type[row.iloc[-2]] = entity_type[tt]

# Test
dftest = pd.read_csv(path_test_path, sep='\t' , header=None)
for i, row in dftest.iterrows():
    s = tokenizer.encode(str(row.iloc[-3])).ids[0]
    t = tokenizer.encode(str(row.iloc[-2])).ids[0]
    r = tokenizer.encode(str(row.iloc[-1])).ids[0]
    st,tt = str(row.iloc[-1]).split('_')[0],str(row.iloc[-1]).split('_')[1]
    if st not in entity_type:
        entity_type[st] = entity_type_num
        entity_type_num+=1
    if tt not in entity_type:
        entity_type[tt] = entity_type_num
        entity_type_num+=1
    node_type[row.iloc[-3]] = entity_type[st]
    node_type[row.iloc[-2]] = entity_type[tt]

# ===============================================
# Construct Train Graph
# ===============================================
s_len =12
g = dgl.DGLGraph()
s, d = dftrain[s_len-2].tolist(),dftrain[s_len-1].tolist()

src = list(zip(s, d))
dst = list(zip(d, s))

src = [item for sublist in src for item in sublist]
dst = [item for sublist in dst for item in sublist]

g.add_edges(src, dst)
g.edata['rel_type'] = torch.LongTensor(rel_type)
g.edata['rel_s_type'] = torch.LongTensor(rel_s_type)
g.edata['rel_s_id'] = torch.LongTensor(rel_s_id)
g.edata['rel_t_type'] = torch.LongTensor(rel_t_type)
g.edata['rel_t_id'] = torch.LongTensor(rel_t_id)

nodes_token_id =[]
nodes_type = [] 
for i in range(g.num_nodes()):
    nodes_token_id.append(get_token_id(i, tokenizer))
    if i in node_type:
        nodes_type.append(node_type[i])
    else:
        nodes_type.append(-1)
        print(i)
g.ndata['token_id'] = torch.LongTensor(nodes_token_id)
g.ndata['node_type'] = torch.LongTensor(nodes_type)

# ===============================================
# Metapath Count
# ===============================================
with open('all_pathcnt_2.pkl', 'rb') as f:
    metapath = pickle.load(f)

s_r_t = {}
for i, row in dftrain.iterrows():
    # print(type(row))
    s = row.iloc[-3]
    t = row.iloc[-2]
    r = row.iloc[-1]
    if s not in s_r_t:
        s_r_t[s] = {}
    if r not in  s_r_t[s]:
        s_r_t[s][r] = set()
    if t not in s_r_t[s][r]:
        s_r_t[s][r].add(t)

for i, row in dftest.iterrows():
    s = row.iloc[-3]
    t = row.iloc[-2]
    r = row.iloc[-1]
    if s not in s_r_t:
        s_r_t[s] = {}
    if r not in  s_r_t[s]:
        s_r_t[s][r] = set()
    if t not in s_r_t[s][r]:
        s_r_t[s][r].add(t)

for i, row in dfdev.iterrows():
    s = row.iloc[-3]
    t = row.iloc[-2]
    r = row.iloc[-1]
    if s not in s_r_t:
        s_r_t[s] = {}
    if r not in  s_r_t[s]:
        s_r_t[s][r] = set()
    if t not in s_r_t[s][r]:
        s_r_t[s][r].add(t)

# ===============================================
# Generate Prompts
# ===============================================

import os
import subprocess
file_path = "MultiHopKG/data/mydata/dev.triples"
script_path = 'source experiment-emb.sh'
arg1 ='configs/mydata.sh'
arg2 ='--inference'
arg3 ='0'
script_code = 'experiment-emb.sh configs/mydata.sh --inference 0'

mode = 'train' # modify for train, dev or test
if mode == 'train':
    df = pd.read_csv(path_train_path, sep='\t' , header=None)
elif mode == 'dev':
    df = pd.read_csv(path_dev_path, sep='\t' , header=None)
else:
    df = pd.read_csv(path_test_path, sep='\t' , header=None)
fpath = f'{mode}_40.txt'

acc =0
sample_size =40
th = 200
possible_ts =[]
st = 0
input_len = 10

with open(fpath, 'w') as f:
    for i, row in df.iterrows():
        s = row.iloc[-3]
        t = row.iloc[-2]
        r = row.iloc[-1]
        r_id = tokenizer.encode(str(row.iloc[-1])).ids[0]
        node_type_t = g.ndata['node_type'][t]
        possible_t = set()
        small_set = []
        large_set = []
        for path_data in list(metapath[r]):
            path = path_data[0]
            next_list =[]
            next_list.append(torch.tensor([s],dtype=torch.int64))
            for j, ent in enumerate(list(path)):
                if len(ent.split("_"))==2:
                    dst_type = entity_type[ent.split("_")[1]]
                else:
                    dst_type = entity_type[ent.split("_")[0]]
                node = next_list[-1]
                _, dst =  g.out_edges(node)
                t_type = g.ndata['node_type'][dst]
                indices = torch.where(t_type == dst_type)
                next_node = torch.unique(dst[indices])
                next_list.append(next_node)

            # print(next_list[-1].size())
            # possible_t = possible_t.union(set(next_list[-1].tolist()))
            set_a =set(next_list[-1].tolist())
            if len(set_a) <th:
                small_set.append(set_a)
            else:
                large_set.append(set_a)

        set_s =set()
        for set_a in small_set:
            set_s = set_s | set_a
        
        if len(large_set)!=0:
            set_l =large_set[0]
            for set_a in large_set:
                set_l = set_l & set_a
            if len(set_s)!=0:
                possible_t = set_s & set_l
            else:
                possible_t = set_l
        else:
            possible_t = set_s

        if t in possible_t: 
            possible_t = possible_t - s_r_t[s][r]
            possible_t.add(t)
            acc+=1
        else:
            possible_t = possible_t -s_r_t[s][r]
        
        possible_ts.append(possible_t)

        # print(possible_t)
        if len(possible_t)!=0:
            # possible_t.add(0)
            repeated_sample = random.choices(list(possible_t), k=sample_size)
            if t in repeated_sample:
                acc+=1
            for sample in repeated_sample:
                f.write(str(sample)+'\t')
        else:
            for j in range(sample_size):
                f.write('DUMMY_ENTITY'+'\t')
        
        for j,sample in enumerate(row):
            if j!=len(row)-1:
                f.write(str(sample)+'\t')
            else:
                f.write(str(sample)+'\n')
                    
        if (i%10000 ==0 and i !=0) or (i == len(df)-1):
            # print(acc, i)
            # Create dev.triples for further filtering
            df[df.columns[-3:]].iloc[st : i].to_csv(
              '/MultiHopKG/data/mydata/dev.triples', sep='\t' , header=None, index=False)
            subprocess.run(["bash", "-c", f"{script_path} {arg1} {arg2} {arg3}"])
            df_temp = pd.read_csv('/MultiHopKG/complex.txt', sep='\t' , header=None)
            for j, row in df_temp.iterrows():
                lst = list(row)
                prompt_metapath =[]
                prompt_complex =list(row)[:20]
                # print(lst)
                for k,nd in enumerate(lst[:-3]):
                    if nd in possible_ts[st+j]:
                        prompt_metapath.append(nd)
                    if len(prompt_metapath) == 20:
                        break
                k = 0
                # print(prompt_metapath,prompt_complex)
                while len(prompt_metapath) < 20:
                    prompt_metapath.append(lst[k])
                    k+=1
                # print(prompt_metapath,prompt_complex)
                for k,sample in enumerate(prompt_metapath):
                        f.write(str(sample)+'\t')
                for k,sample in enumerate(prompt_complex):
                        f.write(str(sample)+'\t')
                f.write(str(df.iloc[st+j,-3])+'\t'+str(df.iloc[st+j,-2])+'\t'+df.iloc[st+j,-1]+'\n')
            st = i+1
            # if i==20:
            #     break
            
    print(acc,i)

