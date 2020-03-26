#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:07:37 2018

@author: function
"""

'''CONFIG SETTINGS FOR NN DEFINITION MODEL

Adapted from original implementation in:
Hill, F., Cho, K., Korhonen, A., & Bengio, Y. (2015). Learning to understand phrases by embedding the
dictionary. arXiv preprint arXiv:1504.00548 .

'''

loss = "cosine" # cosine or rank
LSTM_type = "average" # bidirectional or average

max_seq_len = 150

batch_size = 256
learning_rate = 0.0001
embedding_size = 100
vocab_size = 100000 # orig. vocab size 100000. BPE vocab size will be smaller
num_epochs = 10
data_dir = "data/full_data"
train_file = "training_set_cleaned.tok"
dev_file = "dev_set.tok"
test_file = "test_guardian_600.tok"
restore = True      # Recover a previously trained model 
evaluate = True      # Evaluate on dev set (can be done while tranining)
vocab_file = data_dir + '/definitions_' + str(vocab_size) + '.vocab'
pretrained_target = True
pretrained_input = False

embeddings_path = "data/embeddings/glove_reduced.pickle"

# recommended save directories: models/{LSTM_TYPE}_{BPE OR WORD}
save_dir = "models/avg_word/"
outfile = "outfiles/avg_word.txt"


