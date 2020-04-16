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
from anagen.rsa_model import RNNSpeakerRSAModel, GPTSpeakerRSAModel

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
  parser.add_argument("--from_npy", type=str)
  parser.add_argument("--use_l1", action="store_true")
  parser.add_argument("--s0_model_type", type=str)
  parser.add_argument("--s0_model_path", type=str)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--max_segment_len", type=int, default=512)
  parser.add_argument("--max_num_ctxs_in_batch", type=int, default=8)
  parser.add_argument("--anteced_top_k", type=int, default=5)
  parser.add_argument("--to_npy", type=str)

  args = parser.parse_args()
  # finish adding arguments

  config = util.initialize_from_env(name=args.name)
  model = util.get_model(config)
  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  device = torch.device("cuda" if "GPU" in os.environ and torch.cuda.is_available() else "cpu")

  if args.use_l1:
    if args.s0_model_type in ["rnn", "RNN"]:
      rsa_model = RNNSpeakerRSAModel(args.s0_model_path, args.batch_size,
                                     args.max_segment_len,
                                     args.anteced_top_k,
                                     args.max_num_ctxs_in_batch,
                                     device,
                                     logger=None)
    if args.s0_model_type in ["gpt", "GPT"]:
      rsa_model = GPTSpeakerRSAModel(args.s0_model_path, max_segment_len=80, anteced_top_k=5, device=device)
  else:
    rsa_model = None

  if args.from_npy:
    model.evaluate(None, official_stdout=True, eval_mode=True,
                   to_npy=args.to_npy, from_npy=args.from_npy,
                   rsa_model=rsa_model)
  else:
    with tf.Session() as session:
      model.restore(session)
      # Make sure eval mode is True if you want official conll results
      model.evaluate(session, official_stdout=False, eval_mode=False,
                   to_npy=args.to_npy, from_npy=args.from_npy,
                   rsa_model=rsa_model)
