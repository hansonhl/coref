GPU=0 python rsa_evaluate.py outputs/bert_base.eval_out.npy \
    --conll_eval_path /home/hansonlu/links/data/coref_data/dev.english.v4_gold_conll \
    --use_l1 \
    --batch_size 4 \
    --max_num_ctxs_in_batch 1 \
    --s0_model_type rnn \
    --s0_model_path /home/hansonlu/links/data/rnn_anagen_models/rnn_anagen_0325_basic_batch_768_best.ckpt \
    --s0_normalization full \
    --debug_out_file stdout

# --anagen_model_dir /home/hansonlu/links/data/anagen_models/anagen_anaph_only_b28_lr_default
# GPU=0 python rsa_evaluate.py outputs/bert_base.eval_out.npy \
#   --conll_eval_path /home/hansonlu/links/data/coref_data/dev.english.v4_gold_conll
