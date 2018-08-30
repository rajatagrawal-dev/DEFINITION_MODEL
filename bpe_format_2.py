#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:51:52 2018

@author: function
"""

'''Final part of BPE process, merge preserved heads back with BPE gloss'''

def merge_heads(heads_file, bpe_defs, outfile):
  with open(heads_file, 'r') as f:
    with open(bpe_defs, 'r') as g:
      heads = f.readlines()
      bpe_defs = g.readlines()
      bpe_lines = []
      for line in range(len(heads)):
        bpe_lines.append(heads[line].strip() + ' ' + bpe_defs[line].strip() + '\n')
      with open(outfile, 'w') as f:
        for line in bpe_lines:
          f.write(line)
          
# Output training test and dev sets to bpe/ and full_bpe/ folders
merge_heads('data/processing/original_heads.tok', 'data/processing/original_defs.bpe','data/bpe/training_set.bpe')
merge_heads('data/processing/test_heads.tok', 'data/processing/test_defs.bpe','data/bpe/test_set.bpe')
merge_heads('data/processing/dev_heads.tok', 'data/processing/dev_defs.bpe','data/bpe/dev_set.bpe')

merge_heads('data/processing/full_heads.tok', 'data/processing/full_defs.bpe','data/full_bpe/training_set.bpe')
merge_heads('data/processing/test_heads.tok', 'data/processing/full_test_defs.bpe','data/full_bpe/test_set.bpe')
merge_heads('data/processing/dev_heads.tok', 'data/processing/full_dev_defs.bpe','data/full_bpe/dev_set.bpe')

