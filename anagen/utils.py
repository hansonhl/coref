import numpy as np
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

import json
import torch


def prune_antec_scores(top_antecedents, top_antecedent_scores, top_k):
    idxs = np.argpartition(top_antecedent_scores, top_k, axis=1)[:,-top_k:]
    pruned_top_antecedent_scores = np.take_along_axis(top_antecedent_scores,
                                                      idxs, axis=1)
    idxs = idxs - 1
    pruned_top_antecedents = np.take_along_axis(top_antecedents, idxs, axis=1)

    return pruned_top_antecedents, pruned_top_antecedent_scores
