import os
import pickle
from tokenizers import CharBPETokenizer
import pandas as pd
from transformers import BertConfig, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F
import logging
from transformers import BertConfig, BertModel
from parser_hsf import args

def compute_metrics(rankings):
    h1, h3, h5, h10 = 0, 0, 0, 0
    mrr = 0
    n = len(rankings)
    for ranking in rankings:
        if ranking <= 1:
            h1 += 1
        if ranking <= 3:
            h3 += 1
        if ranking <= 5:
            h5 += 1
        if ranking <= 10:
            h10 += 1
        if ranking <= 256:
            mrr += 1 / ranking
    return h1/n, h3/n, h5/n, h10/n, mrr/n
    # hit_at_k = sum([1 if ranking <= k else 0 for ranking in rankings]) / len(rankings)
    # # mrr = sum([1 / ranking if ranking <= k else 0 for ranking in rankings]) / len(rankings)
    # mrr = sum([1 / ranking if ranking <= 256 else 0 for ranking in rankings]) / len(rankings)
    # return hit_at_k, mrr

class MyModel(nn.Module):
    def __init__(self, embeddings, config, args):
        super().__init__()
        # self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)#nn.Embedding.from_pretrained(embeddings)
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.args = args
        # froze embedding layer
        if not args.finetune:
            for param in self.embedding.parameters():
                param.requires_grad = False
        if args.mask_rate > 0 or args.mixup > 0:
            # create another embedding for mask token use nn.Parameter
            self.mask_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
            # set requires_grad to True
            self.mask_embedding.requires_grad = True
            # initialize with xaiver_normal
            nn.init.xavier_normal_(self.mask_embedding)

        self.encoder = BertModel(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.mask_rate = args.mask_rate

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(dtype=torch.long)
        
        attention_mask = attention_mask.to(dtype=torch.long)

        embedded = self.embedding(input_ids)
        if self.training and self.mask_rate > 0:
            assert(self.mixup == 0)
            # randomly mask input_ids
            mask_tensor = torch.bernoulli(torch.full(input_ids.shape, self.mask_rate)).bool().unsqueeze(-1).to(input_ids.device)
            mask_tensor[:, -2:] = False
            embedded = torch.where(mask_tensor, self.mask_embedding, embedded)
            # embedded = torch.where(mask_tensor, torch.zeros_like(embedded).to(input_ids.device), embedded)

        if self.args.mixup > 0:
            assert(self.mask_rate == 0)
            # bernoulli sample a tensor with shape (batch_size, 1)
            mixup_tensor = torch.bernoulli(torch.full((input_ids.shape[0], 1), self.args.mixup)).bool().to(input_ids.device)
            # repeat the tensor to (batch_size, seq_len)
            mixup_tensor = mixup_tensor.repeat(1, input_ids.shape[1]).unsqueeze(-1)
            mixup_tensor[:, -2:] = False
            embedded = torch.where(mixup_tensor, self.mask_embedding, embedded)   

        outputs = self.encoder(inputs_embeds=embedded, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state.sum(dim=1)

        prediction_scores = self.decoder(sequence_output)

        if not self.training and self.encoder.config.output_attentions:
            return prediction_scores, outputs.attentions

        return prediction_scores
    
class TripletDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length, args):
        self.tokenizer = tokenizer
        self.lines = open(filename, 'r').read().splitlines()
        self.max_length = max_length
        self.args = args
        self.ntokens = len(self.tokenizer.get_vocab())

    def __getitem__(self, idx):
        line = self.lines[idx]
        words = line.split('\t')
        input_str=''
        if self.args.raw:
            input_str = words[input_len] + ' ' + words[input_len+2]
        else:
            for i in range(input_len+1):
                input_str += words[i] +' '
            input_str += words[input_len+2]
        input_ids = torch.tensor(self.tokenizer.encode(input_str).ids)
        labels = torch.tensor(self.tokenizer.encode(words[input_len+1]).ids)
        # pad input_ids and labels
        input_ids = F.pad(input_ids, pad=(0, self.max_length - len(input_ids)))
        if self.args.rand_format:
            perm = torch.randperm(input_ids.size(0))
            input_ids = input_ids[perm]
            input_words = words[:input_len+1] + [words[input_len+2]]
            input_words = [input_words[i] for i in perm]
        else:
            input_words = words[:input_len+1] + [words[input_len+2]]
        labels = F.pad(labels, pad=(0, 1 - len(labels)))

        # create attention_mask
        attention_mask = torch.ones_like(input_ids)
        # print(input_ids.size(),labels.size(),attention_mask.size())
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'qry': (words[input_len], words[input_len+1], words[input_len+2]),
            'input_words': input_words
        }

    def __len__(self):
        return len(self.lines)

def get_false_positive_mask():
    s_r_t = {}
    dftrain = pd.read_csv(args.train_path, delimiter='\t', header=None)
    for i, row in dftrain.iterrows():
        s = tokenizer.encode(str(row[input_len])).ids[0]
        t = tokenizer.encode(str(row[input_len+1])).ids[0]
        r = tokenizer.encode(str(row[input_len+2])).ids[0]
        if s not in s_r_t:
            s_r_t[s]={}
            
        
        if r not in  s_r_t[s]:
            s_r_t[s][r]=set()

        if t not in s_r_t[s][r]:
            s_r_t[s][r].add(t)

    dftest = pd.read_csv(args.test_path, delimiter='\t', header=None)
    for i, row in dftest.iterrows():
        s = tokenizer.encode(str(row[input_len])).ids[0]
        t = tokenizer.encode(str(row[input_len+1])).ids[0]
        r = tokenizer.encode(str(row[input_len+2])).ids[0]
        if s not in s_r_t:
            s_r_t[s]={}
            
        
        if r not in  s_r_t[s]:
            s_r_t[s][r]=set()

        if t not in s_r_t[s][r]:
            s_r_t[s][r].add(t)

    dfdev = pd.read_csv(args.dev_path, delimiter='\t', header=None)
    for i, row in dfdev.iterrows():
        s = tokenizer.encode(str(row[input_len])).ids[0]
        t = tokenizer.encode(str(row[input_len+1])).ids[0]
        r = tokenizer.encode(str(row[input_len+2])).ids[0]
        if s not in s_r_t:
            s_r_t[s]={}
            
        
        if r not in  s_r_t[s]:
            s_r_t[s][r]=set()

        if t not in s_r_t[s][r]:
            s_r_t[s][r].add(t)
    return  s_r_t

def get_token_id(node_id,tokenizer):
    return tokenizer.encode(str(node_id)).ids[0]

def get_id(node_id,tokenizer):
    return int(tokenizer.decode([node_id]))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_len = args.input_len
# accelerator = Accelerator(mixed_precision='fp16')

if args.wandb:
    import wandb
    wandb.init(project="hgc", name=args.name, config=args)
    wandb.config.update(args)

print ("================loading embeddings===================")
entity_embeddings = torch.load(args.entity_embeddings)['weight']
relation_embeddings = torch.load(args.relation_embeddings)['weight']
df_entity2id = pd.read_csv(args.entity2id, sep='\t' , header=None)
df_relation2id = pd.read_csv(args.relation2id, sep='\t' , header=None)
tokenizer = CharBPETokenizer(split_on_whitespace_only=True)
tokenizer.add_tokens(df_entity2id[0].tolist())
tokenizer.add_tokens(df_relation2id[0].tolist())

embeddings = torch.cat((entity_embeddings, relation_embeddings), dim=0)

print ("================get_false_positive_mask==============")
if os.path.exists('./my_data/s_r_t.pkl'):
    s_r_t = pickle.load(open('./my_data/s_r_t.pkl', 'rb'))
else:
    s_r_t = get_false_positive_mask()
    pickle.dump(s_r_t, open('./my_data/s_r_t.pkl', 'wb'))
s_r_t = get_false_positive_mask()

config = BertConfig(
    vocab_size=tokenizer.get_vocab_size(),
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    num_attention_heads=args.num_attention_heads,
    intermediate_size=args.intermediate_size,
)

# print(args)

model = MyModel(embeddings,config, args)
model = model.to(device)
save_path = f'./logs/{args.save_path}'
ckpt_path = f'./logs/{args.save_path}/ckpt'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)
if args.load_model:
    print ("==============loading model==========================")
    # model.load_state_dict(torch.load(args.model_load_path))
    model = torch.load(args.model_load_path)
    logging.basicConfig(level=logging.DEBUG,
                    filename=f'./logs/{args.save_path}/{args.mode}.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
else:
    logging.basicConfig(level=logging.DEBUG,
                    filename=f'./logs/{args.save_path}/{args.mode}.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )    
logging.info(args)


print ("=================loading data========================")
trainDataset = TripletDataset(args.train_path, tokenizer, input_len+2, args)
traindataloader = DataLoader(trainDataset, batch_size=args.batch_size, num_workers=4)

devDataset = TripletDataset(args.dev_path, tokenizer, input_len+2, args)
devdataloader = DataLoader(devDataset, batch_size=args.batch_size, num_workers=4)

testDataset = TripletDataset(args.test_path, tokenizer, input_len+2, args)
testdataloader = DataLoader(testDataset, batch_size=args.batch_size)

optimizer = AdamW(model.parameters(), lr=args.lr)
loss_fct = nn.CrossEntropyLoss()

EPOCHS = args.epochs_num
dummy_mask = [0, 1] 
for i in range(38033 , 38094):
    dummy_mask.append(i)
if args.mode == 'train':
    print ("=================strat train============================")
    # training
    for epoch in range(EPOCHS):
        print(f'Starting epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)
        epoch_loss = 0.0
        best_hit1 = 0.0
        # if i % 100 == 0:
        with torch.no_grad():
            rankings = []
            for test_batch in devdataloader:
                # move to device
                test_input_ids = test_batch['input_ids'].to(device)
                test_attention_mask = test_batch['attention_mask'].to(device)
                test_labels = test_batch['labels'].to(device)
                
                # compute output
                test_outputs = model(test_input_ids, test_attention_mask)
              
                for i, example in enumerate(test_input_ids):
                    e1,  r = int(example[0].cpu()), int(example[1].cpu())
                    e2 = int(test_labels[i][0].cpu())

                    if  (e1 in s_r_t) and (r in s_r_t[e1]):
                         
                        # e2_multi = list(s_r_t[e1][r].union(all_set.difference(type_entity[rel_t[r]]))) 
                        e2_multi = dummy_mask + list(s_r_t[e1][r]) 
                        # save the relevant prediction
                        target_score = float(test_outputs[i, e2])
                        # mask all false negatives
                        # test_outputs[i, e2_multi] = 0
                        test_outputs[i, e2_multi] = -1e5
                        # write back the save prediction
                        test_outputs[i, e2] = target_score

                # compute rankings
                _, predicted_indices = torch.topk(test_outputs, k=test_outputs.size(-1), dim=-1)
                # print(test_labels, predicted_indices)
                for label, predicted_index in zip(test_labels, predicted_indices):

                    ranking = (predicted_index == label).nonzero(as_tuple=True)[0].item() + 1
                    rankings.append(ranking)

            # compute Hit Ratios and MRR
            hit_at_1, hit_at_3, hit_at_5, hit_at_10, mrr = compute_metrics(rankings)
            # round up to 4 decimal places
            hit_at_1 = round(hit_at_1, 4)
            hit_at_3 = round(hit_at_3, 4)
            hit_at_5 = round(hit_at_5, 4)
            hit_at_10 = round(hit_at_10, 4)
            mrr = round(mrr, 4)
            # hit_at_1, mrr = compute_metrics(rankings, k=1)
            # hit_at_10, _ = compute_metrics(rankings, k=10)
            log = f'Hit@1: {hit_at_1}, Hit@3: {hit_at_3}, Hit@5: {hit_at_5}, Hit@10: {hit_at_10}, MRR: {mrr}'
            logging.info(f'Epoch {epoch}/{EPOCHS}: '+log)
            print(log)
            if hit_at_1 > best_hit1:
                best_hit1 = hit_at_1
                torch.save(model, os.path.join(ckpt_path, args.model_save_path))
        
        model.train()
        progress_bar = tqdm(traindataloader)
        cnt = 0
        for i,batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # print(input_ids,attention_mask,labels)
            # print(input_ids.size(), labels.size())

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)

            # print(outputs.size())
            if args.div:
                # convert outputs to softmax probabilities
                outputs = F.softmax(outputs, dim=-1)
                entr_loss = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=-1).mean()
                loss += args.alpha * entr_loss
            if args.label_smooth > 0:
                lprobs = F.log_softmax(outputs, dim=-1)
                loss = -(args.label_smooth * torch.gather(input=lprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze() \
                + (1 - args.label_smooth) / (config.vocab_size - 1) * lprobs.sum(dim=-1))
            else:
                loss = loss_fct(outputs.view(-1, config.vocab_size), labels.view(-1))
          
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar
            cnt += 1
            progress_bar.set_description(f'Loss: {epoch_loss/cnt:.4f}')
        
        # # Print average loss for the epoch
        # avg_loss = epoch_loss / len(traindataloader)
        # print(f'Average Loss: {avg_loss:.4f}')

if args.mode == 'test':
    # ======================== Test ==========================
    model.encoder.config.output_attentions = True
    # (num_layers, batch_size, num_heads, sequence_length, sequence_length)
    attn_scores = torch.zeros((
        args.num_hidden_layers, len(testdataloader.dataset), 
        args.num_attention_heads, input_len+2, input_len+2), dtype=torch.float32)
    qrys = []
    probs = torch.zeros((len(testdataloader.dataset), tokenizer.get_vocab_size()), dtype=torch.float32)
    input_words = []
    # input_ids = torch.zeros((len(testdataloader.dataset), input_len+2), dtype=torch.long)
    rankings = []
    model.eval()
    start = 0
    with torch.no_grad():
        for test_batch in testdataloader:
            # move to device
            test_input_ids = test_batch['input_ids'].to(device)
            test_attention_mask = test_batch['attention_mask'].to(device)
            test_labels = test_batch['labels'].to(device)
            test_input_words = test_batch['input_words']

            # compute outputs
            bsz = test_input_ids.size(0)
            test_outputs, test_attn_scores = model(test_input_ids, test_attention_mask)
            attn_scores[0, start:start+bsz, ...] = test_attn_scores[0].detach().cpu()
            attn_scores[1, start:start+bsz, ...] = test_attn_scores[1].detach().cpu()
            for src, dst, rel in zip(test_batch['qry'][0], test_batch['qry'][1], test_batch['qry'][2]):
                qrys.append((src, dst, rel))
            for i, example in enumerate(test_input_ids):
                e1,  r = int(example[0].cpu()), int(example[1].cpu())
                e2 = int(test_labels[i][0].cpu())

                if  (e1 in s_r_t) and (r in s_r_t[e1]):
                    
                    
                    # e2_multi = list(s_r_t[e1][r].union(all_set.difference(type_entity[rel_t[r]]))) 
                    e2_multi = dummy_mask + list(s_r_t[e1][r]) 
                    # save the relevant prediction
                    target_score = float(test_outputs[i, e2])
                    # mask all false negatives
                    # test_outputs[i, e2_multi] = 0
                    test_outputs[i, e2_multi] = -1e5
                    # write back the save prediction
                    test_outputs[i, e2] = target_score
                
            # compute rankings
            _, predicted_indices = torch.topk(test_outputs, k=test_outputs.size(-1), dim=-1)
            # print(test_labels, predicted_indices)
            for label, predicted_index in zip(test_labels, predicted_indices):

                ranking = (predicted_index == label).nonzero(as_tuple=True)[0].item() + 1
                rankings.append(ranking)
            
            test_outputs = test_outputs.detach().cpu()
            lprobs = F.log_softmax(test_outputs, dim=-1)
            probs[start:start+bsz, :] = lprobs
            # input_ids[start:start+bsz, :] = test_input_ids.detach().cpu()
            for b in range(bsz):
                words = []
                for s in range(input_len+2):  
                    words.append(test_input_words[s][b])
                input_words.append(words)
            start += bsz

        # compute Hit Ratios and MRR
        hit_at_1, hit_at_3, hit_at_5, hit_at_10, mrr = compute_metrics(rankings)
        # round up to 4 decimal places
        hit_at_1 = round(hit_at_1, 4)
        hit_at_3 = round(hit_at_3, 4)
        hit_at_5 = round(hit_at_5, 4)
        hit_at_10 = round(hit_at_10, 4)
        mrr = round(mrr, 4)
        # hit_at_1, mrr = compute_metrics(rankings, k=1)
        # hit_at_10, _ = compute_metrics(rankings, k=10)
        log = f'Hit@1: {hit_at_1}, Hit@3: {hit_at_3}, Hit@5: {hit_at_5}, Hit@10: {hit_at_10}, MRR: {mrr}'
        logging.info(log)
        print(log)

    all_data = {
        'qrys': qrys,
        # 'probs': probs,
        'rankings': rankings,
        # 'attn_scores': attn_scores,
        # 'input_ids': input_ids,
        'input_words': input_words,
    }
    path = os.path.join(save_path, 'all_data.pkl')
    with open(path, 'wb') as f:
        pickle.dump(all_data, f)
    torch.save(probs, os.path.join(save_path, 'probs.pt'))
    torch.save(attn_scores, os.path.join(save_path, 'attn_scores.pt'))
    # torch.save(input_ids, os.path.join(save_path, 'input_ids.pt'))
