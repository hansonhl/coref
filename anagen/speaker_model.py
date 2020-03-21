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
                          num_layers=args.rnn_num_layers,
                          batch_first=True)
        self.null_anteced_emb = nn.Parameter(args.param_init_stdev * torch.randn(self.anteced_emb_size))

    def forward(self, batch):
        anteced_reprs = self.encode(batch)
        pass

    def encode(self, batch):
        ctx_ids = batch["ctx_ids"] # [num_ctx, max_ctx_len]
        ctx_ids_padding_mask = batch["ctx_ids_padding_mask"] # [num_ctx, max_ctx_len]
        ctx_set_idxs = batch["ctx_set_idxs"] # [batch_size]
        anteced_starts = batch["anteced_starts"] # [batch_size]
        anteced_ends = batch["anteced_ends"] # [batch_size]

        gpt2_outputs = self.gpt2_model(ctx_ids,
                                       attention_mask=ctx_ids_padding_mask)

        hidden_states = gpt2_outputs[0] # [num_ctxs, max_ctx_len, hidden_dim]
        flat_hidden_states = torch.cat((self.null_anteced_emb.unsqueeze(0),
            hidden_states.view(-1, hidden_states.shape[-1])),0) # [1+num_ctxs*max_ctx_len, hidden_dim]

        flat_idx_offset = (torch.arange(ctx_ids.shape[0]) * ctx_ids.shape[1])[ctx_set_idxs] # [batch_size]
        flat_anteced_start_idxs = flat_idx_offset + anteced_starts + 1 # [batch_size]
        flat_anteced_end_idxs = flat_idx_offset + anteced_ends + 1 # [batch_size]

        print("hidden_states.shape", hidden_states.shape)
        print("flat_hidden_states.shape", flat_hidden_states.shape)
        print("flat_anteced_start_idxs", flat_anteced_start_idxs)

        is_null = anteced_starts == -1
        flat_anteced_start_idxs[is_null] = 0
        flat_anteced_start_embs = flat_hidden_states.index_select(0, flat_anteced_start_idxs)
        flat_anteced_end_embs = flat_hidden_states.index_select(0, flat_anteced_end_idxs)


        # print("ctx_set_idxs", ctx_set_idxs)
        # print("flat_idx_offset", flat_idx_offset)
        # print("anteced_start_idxs", anteced_starts)
        # print("flat_anteced_start_idxs", flat_anteced_start_idxs)
        print("flat_anteced_start_embs.shape", flat_anteced_start_embs.shape)
        print("flat_anteced_end_embs.shape", flat_anteced_end_embs.shape)
