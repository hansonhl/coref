import torch
import argparse
from anagen.speaker_model import LiteralSpeakerModel
from anagen.dataset import AnagenDataset
from anagen.train import train
from anagen.utils import parse_train_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_train_args(parser)

    # IMPORTANT: must set random seed before initializing model
    if args.random_seed >= 0:
        print("setting random seed to %d" % args.random_seed)
        torch.manual_seed(args.random_seed)

    print("jsonlines_file", args.train_jsonlines)
    train_dataset = AnagenDataset(args.train_jsonlines, args.train_batch_size, args.max_segment_len)
    eval_dataset = AnagenDataset(args.eval_jsonlines, args.eval_batch_size, args.max_segment_len)
    model = LiteralSpeakerModel(args)

    train(args, model, train_dataset, eval_dataset)
