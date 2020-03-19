import torch
import argparse
from anagen.speaker_model import LiteralSpeakerModel
from anagen.dataset import AnagenDataset
from anagen.train import train
from anagen.utils import parse_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    print("jsonlines_file", args.jsonlines_file)
    dataset = AnagenDataset(args.jsonlines_file, args.train_batch_size, args.max_segment_len)
    model = LiteralSpeakerModel(args)

    batch = train(args, dataset, model)

    # check antecedents
    first_batch = dataset.batches[0]
    first_doc = dataset.documents[first.batch[0]]
