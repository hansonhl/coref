import torch
import json
import numpy as np
from anagen.utils import flatten
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

GPT2_EOS_TOKEN_ID = 50256

""" Stores tokenized strings and id form of a document used in the anaphor
    generation task."""
class AnagenDocument:
    def __init__(self, doc_key, segments, segment_starts, speakers, subtoken_map, tokenizer):
        self.doc_key = doc_key
        self.segment_toks = segments
        self.segment_ids = [tokenizer.encode(seg) for seg in segments]
        self.segment_starts = segment_starts
        self.speakers = speakers
        self.subtoken_map = subtoken_map
        self.tokenizer = tokenizer

    """ Given indices ranging in the whole document, obtain corresponding
        segment and index in the segment"""
    def index_in_segments(self, start_idx, end_idx=None):
        # assume that no spans can cross segment boundaries
        seg_idx = np.argmax(self.segment_starts > start_idx) - 1
        start_idx_in_seg = start_idx - self.segment_starts[seg_idx]
        if end_idx is None:
            return seg_idx, start_idx_in_seg
        else:
            end_idx_in_seg = end_idx - self.segment_starts[seg_idx]
            return seg_idx, start_idx_in_seg, end_idx_in_seg

    """ Get the tokens of a span in the current document.
        Args:
            start (int): start idx
            end (int): end idx, inclusive
            in_segment (int or None): segment index, if given then the start
                and end above are relevant to this segment.
            output_str (bool): whether to undo bpe and return original string. """
    def get_span_toks(self, start, end, in_segment=None, output_str=True):
        if start == -1 and end == -1:
            return "<null>"
        if not in_segment:
            in_segment, start, end = self.index_in_segments(start, end)
        span_toks = self.segment_toks[in_segment][start:end+1]
        if output_str:
            return self.tokenizer.convert_tokens_to_string(span_toks)
        else:
            return span_toks

""" Intermediate object representing one training example in the anaphor
    generation task"""
class AnagenExample:
    def __init__(self, doc_key, anteced_start, anteced_end, anaphor_start,
                 anaphor_end, ctx_seg_start_idx, ctx_seg_end_idx):
        self.doc_key = doc_key
        self.anteced_start = anteced_start
        self.anteced_end = anteced_end
        self.anaphor_start = anaphor_start
        self.anaphor_end = anaphor_end
        # indeces of segments that are used to
        self.ctx_seg_start_idx = ctx_seg_start_idx
        self.ctx_seg_end_idx = ctx_seg_end_idx

    def __str__(self):
        return "(Doc \"%s\", segs %d-%d, anteced (%d, %d), anaphor (%d, %d))" \
            % (self.doc_key, self.ctx_seg_start_idx, self.ctx_seg_end_idx,
               self.anteced_start, self.anteced_end,
               self.anaphor_start, self.anaphor_end)

""" Dataset class to input examples for literal speaker of anaphor generation
    task. Takes jsonlines file constructed by anagen_minimize.py as input.
    Initialization:
    [Step 1] _process_jsonline(): reads all lines in the jsonlines file and
        constructs examples and document objects
    [Step 2] _finalize_batches: given example and document objects, construct
        batches of a given batch size. Stores batches in memory

    Sampling batches:
    [Step 1] __getitem__(): takes a batch in memory and retrieves full id form
        of all necessary context sequences. After this step, a batch does not
        contain any reference to document objects.
    [Step 2] collate(batch): converts everything to PyTorch tensors; reorganizes
        examples in order of decreasing anaphor length; adds padding to context
        and anaphor sequences. """
class AnagenDataset(Dataset):
    def __init__(self, jsonlines_file, batch_size, max_num_ctxs_in_batch, max_segment_len=512):
        self.documents = {}
        self.docs_to_examples = {}
        self.batches = []
        self.batch_size = batch_size
        self.max_num_ctxs_in_batch = max_num_ctxs_in_batch
        self.max_segment_len = 512
        self.num_examples = 0
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # just use print for now
        print("Initializing dataset from %s" % jsonlines_file)

        with open(jsonlines_file, "r") as f:
            for line in f:
                self._process_jsonline(line)

        print("Obtained %d examples in total" % self.num_examples)
        print("Compiling batches, batch size %d..." % self.batch_size)
        self._finalize_batches()
        print("Compiled %d batches." % len(self.batches))

        num_examples_in_batches = 0
        for b in self.batches:
            num_examples_in_batches += len(b[2])
        print("Got %d examples in batches, expected %d" % (num_examples_in_batches, self.num_examples))

    """ Get the tokens of a span in a given document.
        See definition in AnagenDocument.get_span_toks()"""
    def get_span_toks(self, doc, start, end, in_segment=None):
        if start == -1 and end == -1:
            return "<null>"
        if isinstance(doc, str):
            return self.documents[doc].get_span_toks(start, end, in_segment)
        if isinstance(doc, AnagenDocument):
            return doc.get_span_toks(start, end, in_segment)
        else:
            ids = doc[start:end+1]
            return self.decode(ids)

    def decode(self, ids):
        # TODO: add function to revert to original word form using subtoken_map.
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    """ Go through jsonlines file and obtain training examples for anaphor generation"""
    def _process_jsonline(self, line):
        coref_example = json.loads(line)
        doc_key = coref_example["doc_key"]
        segments = coref_example["sentences"]
        speakers = coref_example["speakers"]
        subtoken_map = coref_example["subtoken_map"]
        clusters = coref_example["clusters"]

        # includes one extra element to facilitate _get_ctx_seg_idxs().
        segment_lens = np.array([0] + [len(s) for s in segments])
        segment_starts = np.cumsum(segment_lens)

        document = AnagenDocument(doc_key, segments, segment_starts, speakers,
                                  subtoken_map, self.tokenizer)
        self.documents[doc_key] = document

        # get cluster information
        anagen_examples = []
        for cluster in clusters:
            mentions = sorted(tuple(m) for m in cluster)
            for anaphor_i in range(len(mentions)):
                anaphor_start = mentions[anaphor_i][0]
                anaphor_end = mentions[anaphor_i][1]
                ctx_seg_start_idx, ctx_seg_end_idx = \
                    self._get_ctx_seg_idxs(segment_starts, anaphor_start)
                if anaphor_i == 0:
                    # first mention
                    ex = AnagenExample(doc_key,
                                       -1, -1,
                                       anaphor_start, anaphor_end,
                                       ctx_seg_start_idx, ctx_seg_end_idx)
                    anagen_examples.append(ex)
                else:
                    for anteced_i in range(anaphor_i):
                        anteced_start = mentions[anteced_i][0]
                        anteced_end = mentions[anteced_i][1]
                        if anteced_end >= anaphor_start:
                            continue
                        ex = AnagenExample(doc_key,
                                           anteced_start, anteced_end,
                                           anaphor_start, anaphor_end,
                                           ctx_seg_start_idx, ctx_seg_end_idx)
                        anagen_examples.append(ex)

        self.docs_to_examples[doc_key] = anagen_examples
        self.num_examples += len(anagen_examples)


    """ Determine the start of the context"""
    def _get_ctx_seg_idxs(self, segment_starts, anaphor_start):
        ctx_seg_start_idx = np.argmax(anaphor_start - segment_starts <= self.max_segment_len)
        ctx_seg_end_idx = np.argmax(segment_starts > anaphor_start) - 1
        return ctx_seg_start_idx, ctx_seg_end_idx

    """ After preparing all examples, group them into batches stored in memory"""
    def _finalize_batches(self):
        curr_batch = []
        ctxs = set()
        prev_doc_key = None
        for doc_key, examples in self.docs_to_examples.items():
            # ensure all examples in a batch are from the same document
            for example in examples:
                ctxs.add(example.ctx_seg_start_idx)
                if len(curr_batch) >= self.batch_size or len(ctxs) > self.max_num_ctxs_in_batch:
                    self._columnize_and_add_batch(curr_batch, doc_key)
                    curr_batch = [example]
                    ctxs = set([example.ctx_seg_start_idx])
                else:
                    curr_batch.append(example)
            if len(curr_batch) > 0:
                self._columnize_and_add_batch(curr_batch, doc_key)
                curr_batch = []


    """convert batch to "column" form and append batch to self.batches"""
    def _columnize_and_add_batch(self, batch, doc_key):
        # print("***adding a new batch***")
        doc = self.documents[doc_key]

        ctx_starts = [doc.segment_starts[ex.ctx_seg_start_idx] for ex in batch]
        anteced_starts = [ex.anteced_start for ex in batch]
        anteced_ends = [ex.anteced_end for ex in batch]
        anaphor_starts = [ex.anaphor_start for ex in batch]

        # prepare anaphors in id form, and prepare ctx_set:
        # for all ctx ranges (denoted by tuples) in batch, remove duplicates to
        # get a set, and for ranges with same start idx, keep only the longest
        # range.
        anaphor_ids = []
        ctxs = [None for _ in range(len(doc.segment_toks))]
        for ex in batch:
            # print(ex)
            # obtain id form of anaphor, store in memory
            anaphor_seg_idx, start_idx_in_seg, end_idx_in_seg = \
                doc.index_in_segments(ex.anaphor_start, ex.anaphor_end)

            anaphor_id = doc.segment_ids[anaphor_seg_idx][start_idx_in_seg:end_idx_in_seg+1]
            # if isinstance(anaphor_ids, int):
            #     anaphor_ids = [anaphor_ids]
            # if len(batch) == 1:
            #     print("segment_starts", doc.segment_starts)
            #     print("anaphor_seg_idx = %d, start_idx_in_seg = %d, end_idx_in_seg = %d" % (anaphor_seg_idx, start_idx_in_seg, end_idx_in_seg))
            #     print(anaphor_id)
            anaphor_ids.append(anaphor_id)

            ctx_seg_start_idx, ctx_seg_end_idx = ex.ctx_seg_start_idx, ex.ctx_seg_end_idx
            if ctxs[ctx_seg_start_idx] is None or ctxs[ctx_seg_start_idx][1] < ctx_seg_end_idx:
                ctxs[ctx_seg_start_idx] = (ctx_seg_start_idx, ctx_seg_end_idx)
        ctx_set = [ctx for ctx in ctxs if ctx is not None]
        start_idx_to_set_idx = {
            ctx_seg_start_idx: i \
            for i, ctx_seg_start_idx \
            in enumerate([start_i for (start_i, _) in ctx_set])
        }

        # for each item in the batch, get the index of the corresponding context
        # in the set of contexts
        ctx_set_idxs = [start_idx_to_set_idx[ex.ctx_seg_start_idx] for ex in batch]

        # print("doc.segment_starts", doc.segment_starts) #ok
        # print("ctx_set", ctx_set, "ctx_set_idxs", ctx_set_idxs, "ctx_starts", ctx_starts) #ok
        # print("anteced_starts", anteced_starts) #ok
        # print("anteced_ends", anteced_ends) #ok
        # print("anaphor_ids", anaphor_ids) #ok
        #
        # for s, e, anaphor_id in zip(anteced_starts, anteced_ends, anaphor_ids):
            # print("[anteced]", self.get_span_toks(doc, s, e), "[anaphor]", self.decode(anaphor_id))
        # print("len of ctx_set when added to batches", len(ctx_set))
        # print("ctx_set", ctx_set)
        # print("anaphor_ids", anaphor_ids)

        self.batches.append([doc_key, ctx_set, ctx_set_idxs, ctx_starts,
                             anteced_starts, anteced_ends, anaphor_starts, anaphor_ids])

    def __len__(self):
        return len(self.batches)

    """ finish preparation of batch """
    def __getitem__(self, idx):
        doc_key, ctx_set, ctx_set_idxs, ctx_starts, anteced_starts, anteced_ends, \
             anaphor_starts, anaphor_ids = self.batches[idx]
        doc = self.documents[doc_key]

        # obtain full id form of ctx
        ctx_ids = [flatten(doc.segment_ids[start_i:end_i+1]) for start_i, end_i in ctx_set]

        # transform everything else into tensor form.
        # All following tensors have dim [batch_size,]
        ctx_starts = torch.tensor(ctx_starts)
        ctx_set_idxs = torch.tensor(ctx_set_idxs)
        anteced_starts = torch.tensor(anteced_starts)
        anteced_ends = torch.tensor(anteced_ends)
        anaphor_starts = torch.tensor(anaphor_starts)

        anteced_starts_in_ctx = torch.clamp(anteced_starts - ctx_starts, min=-1)
        anteced_ends_in_ctx = torch.clamp(anteced_ends - ctx_starts, min=-1)
        anaphor_starts_in_ctx = anaphor_starts - ctx_starts

        # for ctx_range, ctx_id in zip(ctx_set, ctx_ids): #ok
        #     print("ctx_idx_start_end",ctx_range)
        #     print("[CTX]", self.decode(ctx_id))

        # for s, e, anaphor_id in zip(anteced_starts_in_ctx, anteced_ends_in_ctx, anaphor_ids):
            # print("[anteced]", self.get_span_toks(doc, s, e), "[anaphor]", self.decode(anaphor_id))

        return ctx_ids, ctx_set_idxs, anteced_starts_in_ctx, anteced_ends_in_ctx, \
            anaphor_starts_in_ctx, anaphor_ids

""" Processes data retrieved from Dataset.__getitem__, converts everything into
    tensors. Returns a dictionary containing content of batch:
    'ctx_ids': [num_ctxs, max_ctx_len] id form of context tokens
    'ctx_ids_padding_mask': [num_ctxs, max_ctx_len] mask indicating length
        variation in ctx, to pass into GPT2 model
    'ctx_lens': [num_ctxs,] length of ctxs
    'ctx_set_idxs': [batch_size,] the corresponding ctx for each example
    'anteced_starts': [batch_size,] starting idxs of antecedents, indexed
        into each respective corresponding ctx
    'anteced_ends': [batch_size,] end idxs of antecedents
    'anaphor_ids': [batch_size, max_anaphor_len] id form of gold anaphor tokens
    'anaphor_ids_padding_mask': [batch_size, max_anaphor_len] mask
        indicating length variation in ctx, to pass into GPT2 model
    'anaphor_lens': [batch_size,] lengths of gold anpahors
    'scramble_idxs': [batch_size,] tensor of indices that is applied to sort
        examples in order of decreasing anaphor length."""
def collate(batch):
    ctx_ids, ctx_set_idxs, anteced_starts_in_ctx, anteced_ends_in_ctx, \
        anaphor_starts_in_ctx, anaphor_ids = batch[0]
    ctx_lens = torch.tensor([len(ctx_id) for ctx_id in ctx_ids])

    # just pad ctxs, don't sort them
    ctx_ids = [torch.tensor(ctx) for ctx in ctx_ids] # convert to tensor form
    padded_ctx_ids = torch.nn.utils.rnn.pad_sequence(ctx_ids, batch_first=True,
                                                     padding_value=GPT2_EOS_TOKEN_ID)
    ctx_ids_padding_mask = (torch.arange(padded_ctx_ids.shape[1])[None, :] \
                            < ctx_lens[:, None]).type(torch.FloatTensor)

    # prepare anaphor ids
    anaphor_lens = torch.tensor([len(anaphor_id) for anaphor_id in anaphor_ids])
    sorted_anaphor_lens, scramble_idxs = torch.sort(anaphor_lens, descending=True)
    sorted_anaphor_ids = [torch.tensor(anaphor_ids[i]) for i in scramble_idxs]
    padded_anaphor_ids = torch.nn.utils.rnn.pad_sequence(sorted_anaphor_ids, batch_first=True,
                                                         padding_value=GPT2_EOS_TOKEN_ID)
    anaphor_ids_padding_mask = (torch.arange(padded_anaphor_ids.shape[1])[None, :] \
                                < sorted_anaphor_lens[:, None]).type(torch.FloatTensor)
    # probably don't need this one

    # prepare anteced
    sorted_ctx_set_idxs = ctx_set_idxs[scramble_idxs]
    sorted_anteced_starts = anteced_starts_in_ctx[scramble_idxs]
    sorted_anteced_ends = anteced_ends_in_ctx[scramble_idxs]
    sorted_anaphor_starts = anaphor_starts_in_ctx[scramble_idxs]

    batch_dict = {
        'ctx_ids': padded_ctx_ids, #ok
        'ctx_ids_padding_mask': ctx_ids_padding_mask, #ok
        'ctx_lens': ctx_lens, #ok
        'ctx_set_idxs': sorted_ctx_set_idxs,
        'anteced_starts': sorted_anteced_starts,
        'anteced_ends': sorted_anteced_ends,
        'anaphor_starts': sorted_anaphor_starts,
        'anaphor_ids': padded_anaphor_ids,
        'anaphor_ids_padding_mask': anaphor_ids_padding_mask,
        'anaphor_lens': sorted_anaphor_lens,
        'scramble_idxs': scramble_idxs
    }

    return batch_dict
