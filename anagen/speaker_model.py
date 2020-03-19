import torch
from torch import nn
from transformers import GPT2Model

class LiteralSpeakerModel(nn.Module):
    def __init__(self, args):
        super(LiteralSpeakerModel, self).__init__()
        self.gpt2_model = GPT2Model.from_pretrained(args.gpt2_model_dir if args.gpt2_model_dir else "gpt2")

    def forward(self, batch):
        ctx_ids, anaphor_start_ends, anaphor_ids = batch
        pass
