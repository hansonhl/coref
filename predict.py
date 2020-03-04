from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import util
from anagen_utils import prune_antec_scores

import numpy as np

top_k

if __name__ == "__main__":
  config = util.initialize_from_env()
  log_dir = config["log_dir"]

  # Input file in .jsonlines format.
  input_filename = sys.argv[2]

  # Predictions will be written to this file in .jsonlines format.
  output_filename = sys.argv[3]

  array_output_filename = sys.argv[4]

  model = util.get_model(config)
  saver = tf.train.Saver()

  k_sum = 0
  c_sum = 0
  count = 0

  res = []

  with tf.Session() as session:
    model.restore(session)

    with open(output_filename, "w") as output_file:
      with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
          example = json.loads(line)
          tensorized_example = model.tensorize_example(example, is_training=False)
          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
          _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)

          res_dict = {
              "line": line,
              "top_span_starts": top_span_starts,
              "top_span_ends": top_span_ends,
              "top_antecedents": top_antecedents,
              "top_antecedent_scores": top_antecedent_scores
          }

          top_antecedents, top_antecedent_scores = \
            prune_antec_scores(top_antecedents, top_antecedent_scores, 5)

          res.append(res_dict)

          predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
          example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
          example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
          example['head_scores'] = []

          output_file.write(json.dumps(example))
          output_file.write("\n")
          if example_num % 20 == 0:
            print("Decoded {} examples.".format(example_num + 1))

  with open(array_output_filename, "wb") as f:
      np.save(f, res)
