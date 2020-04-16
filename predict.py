from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import util
import argparse
from tqdm import tqdm

import numpy as np

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model_name", type=str, help="Model name")
  parser.add_argument("input_path", type=str, help="Input file in .jsonlines format")
  parser.add_argument("--output_path", type=str, help="Predictions will be written to this file in .jsonlines format.")
  parser.add_argument("--npy_output_path", type=str, help="Output npy pickle file with model scores")
  args = parser.parse_args()

  model_name = args.model_name
  input_filename = args.input_path
  output_filename = args.output_path
  npy_output_filename = args.npy_output_path

  config = util.initialize_from_env(name=model_name)
  log_dir = config["log_dir"]

  model = util.get_model(config)
  saver = tf.train.Saver()

  k_sum = 0
  c_sum = 0
  count = 0

  if npy_output_filename:
      data_dicts = []

  with tf.Session() as session:
    model.restore(session)

    input_file = open(input_filename, "r")
    if output_filename:
        output_file = open(output_filename, "w")

    for example_num, line in enumerate(tqdm(input_file.readlines())):
      example = json.loads(line)
      tensorized_example = model.tensorize_example(example, is_training=False)
      feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
      _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)

      if npy_output_filename:
          data_dict = {
              "example_num": example_num,
              "example": example,
              "top_span_starts": top_span_starts,
              "top_span_ends": top_span_ends,
              "top_antecedents": top_antecedents,
              "top_antecedent_scores": top_antecedent_scores
          }

          # top_antecedents, top_antecedent_scores = \
          #   prune_antec_scores(top_antecedents, top_antecedent_scores, 5)

          data_dicts.append(data_dict)

      predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
      example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
      example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
      example['head_scores'] = []

      if output_filename:
          output_file.write(json.dumps(example))
          output_file.write("\n")
      # if example_num % 20 == 0:
      #   print("Decoded {} examples.".format(example_num + 1))

    input_file.close()
    if output_filename:
        output_file.close()

  if npy_output_filename:
      dict_to_npy = {"data_dicts": data_dicts}
      with open(npy_output_filename, "wb") as f:
          np.save(f, dict_to_npy)
