GPU=0 python rsa_grid_search.py outputs/bert_base.eval_out.npy \
    --conll_eval_path "/home/hansonlu/links/data/coref_data/dev.english.v4_gold_conll"\
    --csv_save_path "outputs/basic_coarse.csv"\
    --use_l1 \
    --batch_size 64 \
    --max_num_ctxs_in_batch 1 \
    --s0_model_type rnn \
    --s0_model_path /home/hansonlu/links/data/rnn_anagen_models/rnn_anagen_0325_basic_batch_768_best.ckpt \
    --alphas 0.01 0.1 0.2 0.5 0.75 1

# # --anagen_model_dir /home/hansonlu/links/data/anagen_models/anagen_anaph_only_b28_lr_default
# GPU=0 python rsa_evaluate.py outputs/bert_base.eval_out.npy \
#   --conll_eval_path /home/hansonlu/links/data/coref_data/dev.english.v4_gold_conll
