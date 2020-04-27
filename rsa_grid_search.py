import os
import time
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import pickle

import metrics
import conll
from anagen.rsa_model import RNNSpeakerRSAModel, GPTSpeakerRSAModel
from rsa_evaluate import get_predicted_antecedents, evaluate_coref
from collections import defaultdict as DD


def conll_evaluate(l0_inputs, alphas, conll_eval_path, all_top_antecedent_scores):
    print("Compiling clusters and evaluators for conll suite")
    coref_predictions = [{} for _ in alphas]
    coref_evaluators = [metrics.CorefEvaluator() for _ in alphas]
    subtoken_maps = {}

    with open(l0_inputs, "rb") as f:
        data_dicts = np.load(f, allow_pickle=True).item().get("data_dicts")

    for example_num, data_dict in enumerate(tqdm(data_dicts)):
        example = data_dict["example"]
        subtoken_maps[example["doc_key"]] = example["subtoken_map"]
        top_span_starts = data_dict["top_span_starts"]
        top_span_ends = data_dict["top_span_ends"]
        top_antecedents = data_dict["top_antecedents"]

        for i in range(len(alphas)):
            top_antecedent_scores = all_top_antecedent_scores[example["doc_key"]][i]
            predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            coref_predictions[i][example["doc_key"]] = evaluate_coref(top_span_starts,
                top_span_ends, predicted_antecedents, example["clusters"], coref_evaluators[i])

    summary_dict = DD(list)
    for i in range(len(alphas)):
        print("\n*****************************")
        print("******* alpha = %f *******" % alphas[i])
        summary_dict["alpha"].append(alphas[i])
        conll_results = conll.evaluate_conll(conll_eval_path, coref_predictions[i], subtoken_maps, official_stdout=True)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        summary_dict["Average F1 (conll)"].append(average_f1)
        print("Average F1 (conll): {:.2f}%".format(average_f1))

        p,r,f = coref_evaluators[i].get_prf()
        summary_dict["Average F1 (py)"].append(f)
        print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(subtoken_maps.keys())))
        summary_dict["Average precision (py)"].append(p)
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"].append(r)
        print("Average recall (py): {:.2f}%".format(r * 100))

    return summary_dict

def grid_search(l0_inputs, alphas, rsa_model):
    all_top_antecedent_scores = {}

    with open(l0_inputs, "rb") as f:
        data_dicts = np.load(f, allow_pickle=True).item().get("data_dicts")

    for example_num, data_dict in enumerate(tqdm(data_dicts)):
        example = data_dict["example"]

        top_span_starts = data_dict["top_span_starts"]
        top_span_ends = data_dict["top_span_ends"]
        top_antecedents = data_dict["top_antecedents"]
        top_antecedent_scores = data_dict["top_antecedent_scores"]

        top_antecedent_scores = rsa_model.l1(example, top_span_starts, top_span_ends,
                                                 top_antecedents, top_antecedent_scores,
                                                 alphas=alphas)

        all_top_antecedent_scores[example["doc_key"]] = top_antecedent_scores

    return all_top_antecedent_scores

        # for i in range(len(alphas)):
        #     top_antecedent_scores = all_top_antecedent_scores[i]
        #     predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        #     coref_predictions[i][example["doc_key"]] = evaluate_coref(top_span_starts,
        #         top_span_ends, predicted_antecedents, example["clusters"], coref_evaluators[i])

    # print("Ran rsa on %d sentences, avg time per sentence %.2f s" % (num_evaluated, total_time / num_evaluated))



def main():
    # add arguments using argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("l0_inputs", type=str)
    parser.add_argument("--raw_save_path", type=str,
                        help="save model predictions to a pickle file")
    parser.add_argument("--raw_load_path", type=str,
                        help="load model predictions from a pickle file")
    parser.add_argument("--raw_load_path_part2", type=str,
                        help="part 2 of raw_load_path")
    parser.add_argument("--conll_eval_path", type=str,
                        help="conduct suite of conll evaluation metrics by specifying path to file required by conll evaluator")
    parser.add_argument("--csv_save_path", type=str,
                        help="save conll metrics to csv file")
    parser.add_argument("--use_l1", action="store_true")
    parser.add_argument("--s0_model_type", type=str)
    parser.add_argument("--s0_model_path", type=str)

    # arguments for rnn model
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_segment_len", type=int, default=512)
    parser.add_argument("--max_num_ctxs_in_batch", type=int, default=8)
    parser.add_argument("--anteced_top_k", type=int, default=5)

    parser.add_argument("--alphas", type=float, nargs="+")

    args = parser.parse_args()

    if args.raw_load_path:
        print("Loading model predictions from %s" % args.raw_load_path)
        with open(args.raw_load_path, "rb") as f:
            alphas, all_top_antecedent_scores = pickle.load(f)
        if args.raw_load_path_part2:
            with open(args.raw_load_path_part2, "rb") as f:
                _, all_top_antecedent_scores_part2 = pickle.load(f)
            all_top_antecedent_scores.update(all_top_antecedent_scores_part2)
    else:
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

        alphas = list(args.alphas)
        all_top_antecedent_scores = grid_search(args.l0_inputs, alphas=alphas,
                                                rsa_model=rsa_model)

    if args.raw_save_path:
        print("Saving model predictions to %s" % args.raw_save_path)
        with open(args.raw_save_path, "wb") as f:
            pickle.dump((alphas, all_top_antecedent_scores), f)

    if args.conll_eval_path:
        print("Evaluating using conll suite")
        summary_dict = conll_evaluate(args.l0_inputs, alphas,
                                      args.conll_eval_path, all_top_antecedent_scores)
        # summary_dict = conll_evaluate(alphas, args.conll_eval_path,
            # coref_predictions, coref_evaluators, subtoken_maps)
        if args.csv_save_path:
            print("Savimg conll evaluate results to %s" % args.csv_save_path)
            df = pd.DataFrame(summary_dict)
            df.to_csv(args.csv_save_path)



if __name__ == "__main__":
  main()
