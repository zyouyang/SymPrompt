"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Hyperparameters.
"""

import argparse
import os


parser = argparse.ArgumentParser(description='metapath-prompt-transfomer-kg-complete')


parser.add_argument('--train_path',default='my_data/train_40.txt', help='train_path')

parser.add_argument('--test_path',default='my_data/test_40.txt', help='test_path')

parser.add_argument('--dev_path',default='my_data/dev_40.txt', help='dev_path')

parser.add_argument('--entity_embeddings',default='my_data/entity_embeddings.pth', help='entity_embeddings')

parser.add_argument('--relation_embeddings',default='my_data/relation_embeddings.pth', help='relation_embeddings')

parser.add_argument('--entity2id',default='my_data/entity2id.txt', help='entity2id')

parser.add_argument('--relation2id',default='my_data/relation2id.txt', help='relation2id')

parser.add_argument('--load_model',action='store_true',default=False , help='model_load_path')

parser.add_argument('--model_load_path',default='my_model.pth', help='model_load_path')

parser.add_argument('--model_save_path',default='my_model.pth', help='model_save_path')

parser.add_argument('--gpu', type=int, default=0, help='gpu device (default: 0)')

parser.add_argument('--epochs_num',default=1000, type=int, help='epochs_num')

parser.add_argument('--batch_size',default=512, help='batch_size')

parser.add_argument('--lr',default=1e-5, help='learning_rate')

parser.add_argument('--hidden_size',default=200, help='hidden_size')

parser.add_argument('--num_hidden_layers',default=2, help='num_hidden_layers')

parser.add_argument('--num_attention_heads',default=2, help='num_attention_heads')

parser.add_argument('--intermediate_size',default=3072, help='intermediate_size')

parser.add_argument('--mask_rate', default=0.0, type=float, help='input embedding mask rate')

parser.add_argument('--finetune', action='store_true', help='finetune the pretrained embeddings')

# input length
parser.add_argument('--input_len', default=40, type=int, help='input length')

# diversity regulation
parser.add_argument('--div', action='store_true', help='diversity regulation')
parser.add_argument('--alpha', default=0.1, type=float, help='diversity regulation weight')

# mixup
parser.add_argument('--mixup', default=0.0, type=float, help='mixup rate')

# log
parser.add_argument('--save_path', default='model_1', type=str, help='number of your model')
parser.add_argument('--wandb', action='store_true')

# label smoothing
parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing rate')

# mode
parser.add_argument('--mode', default='train', type=str, help='train or test')

# rand format
parser.add_argument('--rand_format', action='store_true', help='random format')

# raw
parser.add_argument('--raw', action='store_true', help='raw')

parser.add_argument('--search_random_seed', action='store_true',
                    help='run experiments with multiple random initializations and compute the result statistics '
                         '(default: False)')

args = parser.parse_args()
