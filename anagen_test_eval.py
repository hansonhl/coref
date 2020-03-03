import os
import torch
import argparse
from anagen.model import CorefRSAModel
from anagen.eval import test_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("name")
    parser.add_argument("--from_npy", type=str)
    parser.add_argument("--use_l1", action="store_true")
    parser.add_argument("--anagen_model_dir", type=str)
    parser.add_argument("--max_segment_len", type=int, default=128)

    args = parser.parse_args()

    device = torch.device("cuda" if "GPU" in os.environ and torch.cuda.is_available() else "cpu")
    rsa_model = CorefRSAModel(args.anagen_model_dir, device, args.max_segment_len)

    test_eval(rsa_model, args)
