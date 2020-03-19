#!/usr/bin/env python

## This file serves as the entry point for using the RSA model ##

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import util
import torch

import argparse
from anagen.model import CorefRSAModel

# to_npy_out_file = "outputs/bert_base.eval_out.npy"

def read_doc_keys(fname):
    keys = set()
    with open(fname) as f:
        for line in f:
            keys.add(line.strip())
    return keys

if __name__ == "__main__":
  # add arguments using argparse
  parser = argparse.ArgumentParser()

  parser.add_argument("name")
  parser.add_argument("--to_npy", type=str)
  parser.add_argument("--from_npy", type=str)
  parser.add_argument("--use_l1", action="store_true")
  parser.add_argument("--anagen_model_dir", type=str)

  args = parser.parse_args()
  # finish adding arguments

  config = util.initialize_from_env(name=args.name)
  model = util.get_model(config)
  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  device = torch.device("cuda" if "GPU" in os.environ and torch.cuda.is_available() else "cpu")

  if args.use_l1:
    rsa_model = CorefRSAModel(args.anagen_model_dir, device=device, max_segment_len=80, anteced_top_k=5)


  with tf.Session() as session:
    model.restore(session)
    # Make sure eval mode is True if you want official conll results
    model.evaluate(session, official_stdout=True, eval_mode=True,
                   to_npy=args.to_npy, from_npy=args.from_npy,
                   rsa_model=rsa_model)
