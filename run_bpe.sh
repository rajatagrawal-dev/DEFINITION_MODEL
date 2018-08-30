# split heads and defs for all train and test sets
python bpe_format_1.py

# learn BPE vocabulary from original training set, 10k merges (can be edited)
echo 'applying BPE to original training set'

./subword-nmt/subword_nmt/learn_bpe.py -s 10000 < data/original_data/definitions_cleaned.tok > data/processing/codes_out

# apply BPE to original defs

./subword-nmt/subword_nmt/apply_bpe.py -c data/processing/codes_out < data/processing/original_defs.tok > data/processing/original_defs.bpe

# apply BPE to dev defs:

./subword-nmt/subword_nmt/apply_bpe.py -c data/processing/codes_out < data/processing/dev_defs.tok > data/processing/dev_defs.bpe

# apply BPE to test defs:

./subword-nmt/subword_nmt/apply_bpe.py -c data/processing/codes_out < data/processing/test_defs.tok > data/processing/test_defs.bpe


# learn BPE vocabulary from full training set, 10k merges (can be edited)
echo 'applying BPE to full training set'
./subword-nmt/subword_nmt/learn_bpe.py -s 10000 < data/full_data/training_set_cleaned.tok > data/processing/codes_out

# apply BPE to full defs

./subword-nmt/subword_nmt/apply_bpe.py -c data/processing/codes_out < data/processing/full_defs.tok > data/processing/full_defs.bpe

# apply BPE to dev defs:

./subword-nmt/subword_nmt/apply_bpe.py -c data/processing/codes_out < data/processing/dev_defs.tok > data/processing/full_dev_defs.bpe

# apply BPE to test defs:

./subword-nmt/subword_nmt/apply_bpe.py -c data/processing/codes_out < data/processing/test_defs.tok > data/processing/full_test_defs.bpe

# merge BPE defs with original heads
python bpe_format_2.py
