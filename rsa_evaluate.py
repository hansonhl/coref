#!/usr/bin/env python

""" This script evaluates the rsa model on the set of standard set of metrics
for coreference resolution. It takes in a pickled npy data file containing the
outputs from the base coref model (l0), which is generated by setting to_npy in
CorefModel.evaluate(). The RSA model is applied to the l0 outputs, and the whole
set of metrics for evaluating coref resolution is run for the l1. This script
contains all the necessary functions to evaluate, and created specifically not
to import tensorflow, in order to save space on the machine running it."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm

import metrics
import conll
from anagen.rsa_model import RNNSpeakerRSAModel, GPTSpeakerRSAModel


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

# copied from CorefModel.get_predicted_antecedents()
def get_predicted_antecedents(antecedents, antecedent_scores):
  predicted_antecedents = []
  for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
    if index < 0:
      predicted_antecedents.append(-1)
    else:
      predicted_antecedents.append(antecedents[i, index])
  return predicted_antecedents

# copied from CorefModel.get_predicted_clusters()
def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
  mention_to_predicted = {}
  predicted_clusters = []
  for i, predicted_index in enumerate(predicted_antecedents):
    if predicted_index < 0:
      continue
    assert i > predicted_index, (i, predicted_index)
    predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
    if predicted_antecedent in mention_to_predicted:
      predicted_cluster = mention_to_predicted[predicted_antecedent]
    else:
      predicted_cluster = len(predicted_clusters)
      predicted_clusters.append([predicted_antecedent])
      mention_to_predicted[predicted_antecedent] = predicted_cluster

    mention = (int(top_span_starts[i]), int(top_span_ends[i]))
    predicted_clusters[predicted_cluster].append(mention)
    mention_to_predicted[mention] = predicted_cluster

  predicted_clusters = [tuple(pc) for pc in predicted_clusters]
  mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

  return predicted_clusters, mention_to_predicted

# copied from CorefModel.evaluate_coref()
def evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
  gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
  mention_to_gold = {}
  for gc in gold_clusters:
    for mention in gc:
      mention_to_gold[mention] = gc

  predicted_clusters, mention_to_predicted = get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
  evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
  return predicted_clusters

def get_l1_scores(l0_inputs, rsa_model, alpha=1.0, debug_out_file=None):
    all_top_antecedent_scores = {}

    with open(l0_inputs, "rb") as f:
        data_dicts = np.load(f, allow_pickle=True).item().get("data_dicts")

    if debug_out_file is not None:
        debug = True
        if debug_out_file == "stdout":
            debug_out_file = None
            print("Starting evaluation, outputting debug info to stdout")
        else:
            print("Starting evaluation, outputting debug info to %s_*.txt" % debug_out_file)
    else:
        debug = None
        print("Starting evaluation, not outputting debug info")
    for example_num, data_dict in enumerate(tqdm(data_dicts)):
        example = data_dict["example"]

        top_span_starts = data_dict["top_span_starts"]
        top_span_ends = data_dict["top_span_ends"]
        top_antecedents = data_dict["top_antecedents"]
        top_antecedent_scores = data_dict["top_antecedent_scores"]

        top_antecedent_scores = rsa_model.l1(example, top_span_starts, top_span_ends,
                                                 top_antecedents, top_antecedent_scores,
                                                 alphas=alpha,
                                                 debug=debug,
                                                 debug_out_file=debug_out_file)

        all_top_antecedent_scores[example["doc_key"]] = top_antecedent_scores

    return all_top_antecedent_scores

# modified from CorefModel.evaluate()
# old evaluate fxn
# def evaluate(l0_inputs, conll_eval_path, alpha=1.0, rsa_model=None, debug_out_file=None):
#   coref_predictions = {}
#   coref_evaluator = metrics.CorefEvaluator()
#   losses = []
#   doc_keys = []
#   num_evaluated = 0
#   total_time = 0
#   subtoken_maps = {}
#
#   with open(l0_inputs, "rb") as f:
#     data_dicts = np.load(f, allow_pickle=True).item().get("data_dicts")
#
#   for example_num, data_dict in enumerate(data_dicts):
#     example = data_dict["example"]
#
#     doc_key = example["doc_key"]
#     subtoken_map = example["subtoken_map"]
#     doc_keys.append(doc_key)
#     subtoken_maps[doc_key] = subtoken_map
#
#     tensorized_example = data_dict["tensorized_example"]
#     loss = data_dict["loss"]
#     top_span_starts = data_dict["top_span_starts"]
#     top_span_ends = data_dict["top_span_ends"]
#     top_antecedents = data_dict["top_antecedents"]
#     top_antecedent_scores = data_dict["top_antecedent_scores"]
#
#     losses.append(loss)
#
#     if rsa_model is not None:
#       start_time = time.time()
#       if debug_out_file is not None:
#         debug = True
#         if debug_out_file == "stdout":
#           debug_out_file = None
#       else:
#         debug = None
#       top_antecedent_scores = rsa_model.l1(example, top_span_starts, top_span_ends,
#                                            top_antecedents, top_antecedent_scores,
#                                            alphas=alpha,
#                                            debug=debug,
#                                            debug_out_file=debug_out_file)
#       duration = time.time() - start_time
#       total_time += duration
#       num_evaluated += 1
#
#     predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)
#     coref_predictions[example["doc_key"]] = evaluate_coref(top_span_starts,
#       top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator)
#
#     if example_num % 10 == 0:
#       print("Evaluated %d/%d examples. Avg time per sentence %.2f s." % \
#             (example_num + 1, len(data_dicts), total_time / num_evaluated))
#
#   if rsa_model:
#     print("Ran rsa on %d sentences, avg time per sentence %.2f s" % (num_evaluated, total_time / num_evaluated))
#
#   summary_dict = {}
#
#   conll_results = conll.evaluate_conll(conll_eval_path, coref_predictions, subtoken_maps, official_stdout=True)
#   average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
#   summary_dict["Average F1 (conll)"] = average_f1
#   print("Average F1 (conll): {:.2f}%".format(average_f1))
#
#   p,r,f = coref_evaluator.get_prf()
#   summary_dict["Average F1 (py)"] = f
#   print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
#   summary_dict["Average precision (py)"] = p
#   print("Average precision (py): {:.2f}%".format(p * 100))
#   summary_dict["Average recall (py)"] = r
#   print("Average recall (py): {:.2f}%".format(r * 100))
#
#   return summary_dict

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

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_segment_len", type=int, default=512)
    parser.add_argument("--max_num_ctxs_in_batch", type=int, default=8)
    parser.add_argument("--anteced_top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.)

    parser.add_argument("--s0_normalization", type=str, default="length")
    parser.add_argument("--debug_out_file", type=str)

    args = parser.parse_args()
    # finish adding arguments

    device = torch.device("cuda" if "GPU" in os.environ and torch.cuda.is_available() else "cpu")


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
        print("alpha:", args.alpha)

        device = torch.device("cuda" if "GPU" in os.environ and torch.cuda.is_available() else "cpu")

        if args.use_l1:
            if args.s0_model_type in ["rnn", "RNN"]:
                rsa_model = RNNSpeakerRSAModel(args.s0_model_path,
                                            batch_size=args.batch_size,
                                            max_segment_len=args.max_segment_len,
                                            anteced_top_k=args.anteced_top_k,
                                            max_num_ctxs_in_batch=args.max_num_ctxs_in_batch,
                                            device=device,
                                            s0_normalization=args.s0_normalization,
                                            logger=None)
            if args.s0_model_type in ["gpt", "GPT"]:
                rsa_model = GPTSpeakerRSAModel(args.s0_model_path,
                                            max_segment_len=args.max_segment_len,
                                            anteced_top_k=args.anteced_top_k,
                                            device=device)
        else:
            rsa_model = None

        all_top_antecedent_scores = get_l1_scores(args.l0_inputs, alpha=args.alpha,
                                                  rsa_model=rsa_model,
                                                  debug_out_file=args.debug_out_file)
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
            print("Saving conll evaluate results to %s" % args.csv_save_path)
            df = pd.DataFrame(summary_dict)
            df.to_csv(args.csv_save_path)

if __name__ == "__main__":
  main()
