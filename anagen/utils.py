import numpy as np
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

import json
import torch
import argparse

def prune_antec_scores(top_antecedents, top_antecedent_scores, top_k):
    idxs = np.argpartition(top_antecedent_scores, top_k, axis=1)[:,-top_k:]
    pruned_top_antecedent_scores = np.take_along_axis(top_antecedent_scores,
                                                      idxs, axis=1)
    idxs = idxs - 1
    pruned_top_antecedents = np.take_along_axis(top_antecedents, idxs, axis=1)

    return pruned_top_antecedents, pruned_top_antecedent_scores

# copied from util.flatten()
def flatten(l):
  return [item for sublist in l for item in sublist]

def parse_args(parser):
    # data input
    parser.add_argument("--jsonlines_file", type=str)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--max_segment_len", type=int, default=512)

    # gpt2 model settings
    parser.add_argument("--gpt2_model_dir", type=str, default=None)

    # training settings
    parser.add_argument("--random_seed", type=int, default=39)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_train_epochs", type=int, default=1)

    # model settings
    parser.add_argument("--gpt2_hidden_size", type=int, default=768)
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--param_init_stdev", type=float, default=0.1)
    parser.add_argument("--rnn_num_layers", type=int, default=1)


    return parser.parse_args()
