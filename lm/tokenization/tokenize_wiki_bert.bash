DATA_DIR=/dataset/wiki/
TOKENIZER=bert-base-uncased
python3 tokenization/tokenize_dataset.py $DATA_DIR en.valid.raw $TOKENIZER
python3 tokenization/tokenize_dataset.py $DATA_DIR en.test.raw $TOKENIZER
python3 tokenization/tokenize_dataset.py $DATA_DIR en.train.raw $TOKENIZER