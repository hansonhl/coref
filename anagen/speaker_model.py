import torch
from torch import nn
from transformers import GPT2Model

class LiteralSpeakerModel(nn.Module):
    def __init__(self, args):
        super(LiteralSpeakerModel, self).__init__()
        self.gpt2_model = GPT2Model.from_pretrained(args.gpt2_model_dir \
            if args.gpt2_model_dir else "gpt2")

        self.anteced_emb_size = args.gpt2_hidden_size
        self.ctx_emb_size = args.gpt2_hidden_size
        self.rnn_hidden_size = self.anteced_emb_size + self.ctx_emb_size

        if args.use_metadata:
            self.rnn_hidden_size += args.anteced_len_emb_size + args.distance_emb_size

        self.rnn = nn.GRU(input_size=args.gpt2_hidden_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=args.num_layers,
                          batch_first=True)
        self.null_anteced_emb = nn.Parameter(args.param_init_stdev * torch.randn(self.anteced_emb_size))

    def forward(self, batch):
        anteced_reprs = self.encode(batch)
        pass

    def encode(self, batch):
        ctx_ids = batch["ctx_ids"]
        ctx_ids_padding_mask = batch["ctx_ids_padding_mask"]

        gpt2_outputs = self.gpt2_model(ctx_ids,
                                       attention_mask=ctx_ids_padding_mask)

        hidden_states = gpt2_outputs[0] # [num_ctxs, max_ctx_len, hidden_dim]
        anteced

        print("hidden_states.shape", hidden_states.shape)
