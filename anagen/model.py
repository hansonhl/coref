import torch
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer

class CorefRSAModel:
    def __init__(self, model_dir, device, max_segment_len):
        # self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        # model = GPT2LMHeadModel.from_pretrained(model_dir)
        # model.to(device)
        # model.eval()
        # self.model = model
        self.max_segment_len = max_segment_len
        pass

    def get_sentence_starts(self, sentence_map):
        starts = [0]
        curr = 0
        for i in range(len(sentence_map)):
            if sentence_map[i] != curr:
                starts.append(i)
                curr = sentence_map[i]
        return starts

    def flatten_sentences(self, sentences):
        res = []
        for s in sentences:
            res += s
        return res

    def get_ctx_start(self, sentence_starts, anaphor_start):
        for i in range(len(sentence_starts)):
            if sentence_starts[i] > anaphor_start:
                anaphor_sent_idx = i-1 # i cannot be 0 in this case, as anaphor_start >= 0 and sentence_starts[0] == 0
                break

        ctx_start_sent_idx = 0
        while anaphor_start - sentence_starts[ctx_start_sent_idx] > self.max_segment_len:
            ctx_start_sent_idx += 1
        return sentence_starts[ctx_start_sent_idx]



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
        """
        print(top_span_starts.shape)
        print(top_span_ends.shape)
        print(top_antecedents.shape)
        print("Find first mention with non-null antec")
        #for i in range(top_antecedents.shape[0]):
        i = 140
        anaphor_start = top_span_starts[i]
        anaphor_end = top_span_ends[i]
        # anteced_arr_idx = np.argmax(top_antecedent_scores[i]) - 1

        #anteced_span_idx = top_antecedents[i][anteced_arr_idx]
        #anteced_start = int(top_span_starts[anteced_span_idx])
        #anteced_end = int(top_span_ends[anteced_span_idx])
        print("anaphor: (%d, %d)" % (anaphor_start, anaphor_end))
        # print("anteced: (%d, %d)" %  (anteced_start, anteced_end))

        sentence_starts = self.get_sentence_starts(example["sentence_map"])
        ctx_start = self.get_ctx_start(sentence_starts, anaphor_start)
        print("ctx_start:", ctx_start)

        flattened_sentences = self.flatten_sentences(example["sentences"])
        print("[ctx in strings]\n", " ".join(flattened_sentences[ctx_start:anaphor_start]))
        print("[anaphor in strings]\n", " ".join(flattened_sentences[anaphor_start:anaphor_end]))

        ## obtain string form

        example_sentence_map = example["sentence_map"]


        max_end = top_span_ends[-1]
        # print("Max end:", max_end)
        # print("Sentence map:", example_sentence_map)
        # print("Sentence map len:", len(example_sentence_map))
        # print("flattened_sentneces:", flattened_sentences)
        # print("flattened_sentences_len:", len(flattened_sentences))


        anaphor_sent = example_sentence_map[anaphor_start]
        anaphor_sent2 = example_sentence_map[anaphor_end]
        print("anaphor_sent check: %d, %d" % (anaphor_sent, anaphor_sent2))

        # check if any top_span_starts or ends contain [CLS] or [sep]
        for i in range(len(top_span_starts)):
            start = top_span_starts[i]
            end = top_span_ends[i]
            if flattened_sentences[start] == "[CLS]":
                print("Start %d is [CLS]!" % start)
            if flattened_sentences[start-1] == "[CLS]":
                print("Start %d-1 is [CLS]!" % start)
            if flattened_sentences[start] == "[SEP]":
                print("Start %d is [SEP]!" % start)
            if flattened_sentences[end] == "[CLS]":
                print("End %d is [CLS]!" % end)
            if flattened_sentences[end] == "[SEP]":
                print("End %d is [SEP]!" % end)
            if flattened_sentences[end+1] == "[SEP]":
                print("End %d+1 is [SEP]!" % end)



        # generate input to model
