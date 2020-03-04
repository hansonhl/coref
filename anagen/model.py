import torch
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer

class CorefRSAModel:
    def __init__(self, model_dir, device, max_segment_len, anteced_top_k):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        self.model = model
        self.device = device
        self.max_segment_len = max_segment_len
        self.anteced_top_k = anteced_top_k

    def get_sentence_starts(self, sentence_map):
        starts = [0]
        curr = 0
        for i in range(len(sentence_map)):
            if sentence_map[i] != curr:
                starts.append(i)
                curr = sentence_map[i]
        starts = np.array(starts)
        return starts

    def flatten_sentences(self, sentences):
        res = []
        for s in sentences:
            res += s
        return res

    def get_ctx_start(self, sentence_starts, anaphor_start, anteced_start):
        anaphor_sent_idx = np.argmax(sentence_starts > anaphor_start) - 1
        if anteced_start is not None:
            anteced_sent_idx = np.argmax(sentence_starts > anteced_start) - 1

        if anteced_start is None or anaphor_start - sentence_starts[anteced_sent_idx] <= self.max_segment_len:
            # make ctx as long as possible under max segment len
            ctx_start_sent_idx = np.argmax(anaphor_start - sentence_starts <= self.max_segment_len)
        else:
            # anteced is very far from anaphor
            ctx_start_sent_idx = anteced_sent_idx
        return sentence_starts[ctx_start_sent_idx]

    def convert_tokens_to_string(self, tokens):
        # filter [CLS] and [SEP]
        tokens = list(filter(lambda x: x != "[CLS]" and x != "[SEP]", tokens))

        res = " ".join(tokens) \
                 .replace(" ##", "") \
                 .replace(" [CLS] ", " ") \
                 .replace(" [SEP] ", " ") \
                 .strip()
        res += " <anaphor>"
        return res

    def top_k_idxs_along_axis1(self, a):
        idxs = np.argpartition(a, -self.anteced_top_k, axis=1)[:,-self.anteced_top_k:]
        return idxs

    def get_single_l1_score(self, input_str, anaphor_str):
        # based off of anagen/evaluate.py
        input_toks = self.tokenizer.encode(input_str)
        anaphor_toks = self.tokenizer.encode(anaphor_str)
        lh_context = torch.tensor(input_toks, dtype=torch.long, device=self.device)
        generated = lh_context

        with torch.no_grad():
            inputs = {"input_ids": generated}
            outputs = self.model(**inputs)
            print(outputs[0].shape)
            next_token_logits = outputs[0][-1, :]

    def get_s0_scores(self, input_strs, anaphor_strs):
        # based off of anagen/evaluate.py
        for


    def l1(self, example, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores):
        print("********** Running l1 ***********")
        """
        print(type(example))
        print(example.keys())
        print("***** doc_key")
        print(example["doc_key"])
        print("***** sentences")
        print(example["sentences"])
        print("***** speakers")
        print(example["speakers"])
        print("***** constituents")
        print(example["constituents"])
        print("***** ner")
        print(example["ner"])
        print("***** clusters")
        print(example["clusters"])
        print("***** sentence_map")
        print(example["sentence_map"])
        print("***** subtoken_map")
        print(example["subtoken_map"])
        print("***** pronouns")
        print(example["pronouns"])
        print("***** top_antecedents")
        print(top_antecedents)
        print("***** top_antecedent_scores")
        print(top_antecedent_scores)

        print(top_span_starts.shape)
        print(top_span_ends.shape)
        print(top_antecedents.shape)
        print("Find first mention with non-null antec")
        """
        # get top k antecedents
        all_anteced_arr_idxs = self.top_k_idxs_along_axis1(top_antecedent_scores)
        # get span indeces for each one, null anteced has span idx -1.
        all_anteced_span_idxs = np.where(all_anteced_arr_idxs != 0,
            np.take_along_axis(top_antecedents, all_anteced_arr_idxs-1, axis=1), -1)

        # get bert-style tokens in one single list
        raw_bert_toks = self.flatten_sentences(example["sentences"])
        # get starting positions of sentences
        sentence_starts = self.get_sentence_starts(example["sentence_map"])

        for anaphor_span_idx in range(top_antecedents.shape[0]):
            anaphor_start = top_span_starts[anaphor_span_idx]
            anaphor_end = top_span_ends[anaphor_span_idx]
            anteced_span_idxs = all_anteced_span_idxs[anaphor_span_idx]
            print("*******************")
            print("anaphor %d: (%d, %d) %s" % (anaphor_span_idx, anaphor_start, anaphor_end, " ".join(raw_bert_toks[anaphor_start:anaphor_end+1])))
            anaphor_toks = raw_bert_toks[anaphor_start:anaphor_end+1]
            anaphor_str = self.convert_tokens_to_string(anaphor_toks)

            all_input_strs = []
            for anteced_span_idx in anteced_span_idxs:
                if anteced_span_idx >= anaphor_span_idx:
                    anteced_start = int(top_span_starts[anteced_span_idx])
                    anteced_end = int(top_span_ends[anteced_span_idx])
                    print("  invalid anteced %d: (%d, %d) %s" % (anteced_span_idx, anteced_start, anteced_end, " ".join(raw_bert_toks[anteced_start:anteced_end+1])))
                    continue

                elif anteced_span_idx >= 0:
                    # assume arr idx is the correct ordering of spans by
                    # start token, and then by length
                    anteced_start = int(top_span_starts[anteced_span_idx])
                    anteced_end = int(top_span_ends[anteced_span_idx])
                    ctx_start = self.get_ctx_start(sentence_starts, anaphor_start, anteced_start)
                    print("  anteced %d: (%d, %d) %s" % (anteced_span_idx, anteced_start, anteced_end, " ".join(raw_bert_toks[anteced_start:anteced_end+1])))
                else:
                    ctx_start = self.get_ctx_start(sentence_starts, anaphor_start, None)
                    print("  Null anteced")

                ctx_tokens = raw_bert_toks[ctx_start:anaphor_start]

                if anteced_span_idx >= 0 and anteced_span_idx < anaphor_span_idx \
                    and anteced_end < anaphor_start:
                    # currently ignore nested anaphors
                    ctx_tokens.insert(anteced_start - ctx_start, "<anteced>")
                    ctx_tokens.insert(anteced_end - ctx_start + 2, "</anteced>")
                else:
                    # temporary measure to indicate null anteced
                    ctx_tokens = ["<anteced>", "</anteced>"] + ctx_tokens

                input_str = self.convert_tokens_to_string(ctx_tokens)
                all_input_strs.append(input_str)

                # feed into GPT model to get probabilities
            scores = self.get_single_l1_score(input_str, anaphor_str)
            return



            if anaphor_span_idx == 10:
                return
