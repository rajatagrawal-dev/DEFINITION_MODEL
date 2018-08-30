#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:44:57 2018

@author: function
"""

'''Formatting data for BPE scripts'''

def split_heads(file, head_file, def_file):
  # split the file to preserve the head words
  head_words = []
  definitions = []
  with open(file, 'r') as f:
    lines = f.readlines()
    
  for line in lines:
    head_words.append(line.split()[0])
    definitions.append(line.split()[1:])
  
  with open(head_file, 'w') as f:
    f.write('\n'.join(head_words))
  
  with open(def_file, 'w') as f:
    for line in definitions:
      for word in line:
        f.write(word + ' ')
      f.write('\n')

# split heads for original training data, dev and test sets
split_heads('data/original_data/definitions_cleaned.tok', 'data/processing/original_heads.tok', 'data/processing/original_defs.tok')
split_heads('data/original_data/dev_set.tok', 'data/processing/dev_heads.tok', 'data/processing/dev_defs.tok')
split_heads('data/original_data/test_set.tok', 'data/processing/test_heads.tok', 'data/processing/test_defs.tok')

# split heads for full training data (dev and test are the same for all experiments)
split_heads('data/full_data/training_set_cleaned.tok', 'data/processing/full_heads.tok', 'data/processing/full_defs.tok')

