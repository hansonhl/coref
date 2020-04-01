import os
import time
import torch
import numpy as np
import pandas as pd
import argparse
import tqdm

import metrics
import conll
from anagen.rsa_model import RNNSpeakerRSAModel, GPTSpeakerRSAModel
from rsa_evaluate import evaluate, get_predicted_antecedents, evaluate_coref
from collections import defaultdict as DD

def grid_search(l0_inputs, conll_eval_path, alphas, rsa_model):
    coref_predictions = [{} for _ in alphas]
    coref_evaluators = [metrics.CorefEvaluator() for _ in alphas]
    doc_keys = []
    num_evaluated = 0
    total_time = 0
    subtoken_maps = {} # for conll evaluator

    with open(l0_inputs, "rb") as f:
        data_dicts = np.load(f, allow_pickle=True).item().get("data_dicts")

    for example_num, data_dict in enumerate(data_dicts):
        example = data_dict["example"]

        doc_key = example["doc_key"]
        subtoken_map = example["subtoken_map"]
        doc_keys.append(doc_key)
        subtoken_maps[doc_key] = subtoken_map

        tensorized_example = data_dict["tensorized_example"]
        top_span_starts = data_dict["top_span_starts"]
        top_span_ends = data_dict["top_span_ends"]
        top_antecedents = data_dict["top_antecedents"]
        top_antecedent_scores = data_dict["top_antecedent_scores"]

        start_time = time.time()
        all_top_antecedent_scores = rsa_model.l1(example, top_span_starts, top_span_ends,
                                                 top_antecedents, top_antecedent_scores,
                                                 alphas=alphas)
        duration = time.time() - start_time
        total_time += duration
        num_evaluated += 1

        for i in range(len(alphas)):
            top_antecedent_scores = all_top_antecedent_scores[i]
            predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            coref_predictions[i][example["doc_key"]] = evaluate_coref(top_span_starts,
                top_span_ends, predicted_antecedents, example["clusters"], coref_evaluators[i])

        if example_num % 50 == 0:
            print("Evaluated {}/{} examples.".format(example_num + 1, len(data_dicts)))

    print("Ran rsa on %d sentences, avg time per sentence %.2f s" % (num_evaluated, total_time / num_evaluated))

    summary_dict = DD(list)
    for i in range(len(alphas)):
        summary_dict["alpha"].append(alphas[i])
        conll_results = conll.evaluate_conll(conll_eval_path, coref_predictions[i], subtoken_maps, official_stdout=True)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        summary_dict["Average F1 (conll)"].append(average_f1)
        # print("Average F1 (conll): {:.2f}%".format(average_f1))

        p,r,f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"].append(f)
        # print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
        summary_dict["Average precision (py)"].append(p)
        # print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"].append(r)
        # print("Average recall (py): {:.2f}%".format(r * 100))

    return summary_dict


def main():
    # add arguments using argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("l0_inputs", type=str)
    parser.add_argument("--conll_eval_path", type=str, required=True)
    parser.add_argument("--csv_save_path", type=str)
    parser.add_argument("--use_l1", action="store_true")
    parser.add_argument("--s0_model_type", type=str)
    parser.add_argument("--s0_model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_segment_len", type=int, default=512)
    parser.add_argument("--max_num_ctxs_in_batch", type=int, default=8)
    parser.add_argument("--anteced_top_k", type=int, default=5)
    parser.add_argument("--alphas", type=float, nargs="+")

    args = parser.parse_args()
    # finish adding arguments
    print("alphas:", args.alphas)

    device = torch.device("cuda" if "GPU" in os.environ and torch.cuda.is_available() else "cpu")

    if args.use_l1:
        if args.s0_model_type in ["rnn", "RNN"]:
            rsa_model = RNNSpeakerRSAModel(args.s0_model_path,
                                           batch_size=args.batch_size,
                                           max_segment_len=args.max_segment_len,
                                           anteced_top_k=args.anteced_top_k,
                                           max_num_ctxs_in_batch=args.max_num_ctxs_in_batch,
                                           device=device,
                                           logger=None)
        if args.s0_model_type in ["gpt", "GPT"]:
            rsa_model = GPTSpeakerRSAModel(args.s0_model_path,
                                           max_segment_len=args.max_segment_len,
                                           anteced_top_k=args.anteced_top_k,
                                           device=device)
    else:
        rsa_model = None

    summary_dict = grid_search(args.l0_inputs, args.conll_eval_path,
                               alphas=list(args.alphas),
                               rsa_model=rsa_model)
    df = pd.DataFrame(summary_dict)
    if args.csv_save_path:
        df.to_csv(args.csv_save_path)



if __name__ == "__main__":
  main()
