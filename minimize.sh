
vocab_file=cased_config_vocab/vocab.txt
data_dir=/home/hansonlu/links/data/coref_data

python minimize.py $vocab_file $data_dir $data_dir false
