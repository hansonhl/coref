import numpy as np

def count_ratio():
    with open("outputs/dev.english.128.out.npy", "rb") as f:
        res = np.load(f)

    total_mentions = 0
    null_antec_mentions = 0

    for example_dict in res:
        antec_scores = example_dict["top_antecedent_scores"];
        max_score = antec_scores.max(1)
        non_zero = np.count_nonzero(max_score)
        total_mentions += max_score.shape[0]
        null_antec_mentions += max_score.shape[0] - non_zero

    print("null antec mentions:", null_antec_mentions)
    print("total mentions:", total_mentions)
    print("ratio: ", null_antec_mentions / total_mentions)

def test_eval(rsa_model, args):
    with open(args.from_npy, "rb") as f:
        from_npy_dict = np.load(f)
        data_dicts = from_npy_dict.item().get("data_dicts")

    for example_num, data_dict in enumerate(data_dicts):
        example = data_dict["example"]
        tensorized_example = data_dict["tensorized_example"]
        loss = data_dict["loss"]
        top_span_starts = data_dict["top_span_starts"]
        top_span_ends = data_dict["top_span_ends"]
        top_antecedents = data_dict["top_antecedents"]
        top_antecedent_scores = data_dict["top_antecedent_scores"]

        rsa_model.l1(example, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores)
        if example_num == 0:
            return
