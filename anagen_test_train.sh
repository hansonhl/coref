python anagen_test_train.py \
    --jsonlines_file data/dev.english.256.twodoc.anagen.jsonlines \
    --train_batch_size 16 \
    --gpu \
    --num_train_epochs 40 \
    --eval_and_save_steps 10
