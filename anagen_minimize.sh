tokenizer_dir="/home/hansonlu/links/data/anagen_models/anagen_anaph_only_b28_lr_default"
input_dir="/home/hansonlu/links/data/coref_data"
output_dir="/home/hansonlu/links/data/pp_coref_anagen"

python anagen_minimize.py $tokenizer_dir $input_dir $output_dir 128
