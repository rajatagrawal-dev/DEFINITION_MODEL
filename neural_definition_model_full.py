"""Language encoder for dictionary definitions.

For training, takes (target-word, dictionary-definition) pairs and
optimises the encoder to produce a single vector for each definition
which is close to the vector for the corresponding target word.

The definitions encoder can be either a bag-of-words or an RNN model.

The vectors for the target words, and the words making up the
definitions, can be either pre-trained or learned as part of the
training process.

Sometimes the definitions are referred to as "glosses", and the target
words as "heads".

Inspiration from Tensorflow documentation (www.tensorflow.org).
The data reading functions were taken from
https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

JP: Adapted from original implementation in:
Hill, F., Cho, K., Korhonen, A., & Bengio, Y. (2015). Learning to understand phrases by embedding the
dictionary. arXiv preprint arXiv:1504.00548 

There are some significant differences from the original implementation, but 
much of the code remains the same. For reference the original is included in this repo

Every endeavour is taken in this code to indicate which parts are my own work.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import sys

import numpy as np
import scipy.spatial.distance as dist
import tensorflow as tf
import pandas as pd

import data_utils_BPE

import config

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
try:
  del_all_flags(tf.flags.FLAGS)
except AttributeError:
  pass



tf.app.flags.DEFINE_integer("max_seq_len", config.max_seq_len, "Maximum length (in words) of a"
                            "definition processed by the model")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "batch size")
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate,
                          "Learning rate applied in TF optimiser")
tf.app.flags.DEFINE_integer("embedding_size", config.embedding_size,
                            "Number of units in word representation.")
tf.app.flags.DEFINE_integer("vocab_size", config.vocab_size, "Nunber of words the model"
                            "knows and stores representations for")
tf.app.flags.DEFINE_integer("num_epochs", config.num_epochs, "Train for this number of"
                            "sweeps through the training set")
tf.app.flags.DEFINE_string("data_dir", config.data_dir, "Directory for finding"
                           "training data and dumping processed data.")
tf.app.flags.DEFINE_string("train_file", config.train_file,
                           "File with dictionary definitions for training.")
tf.app.flags.DEFINE_string("dev_file", config.dev_file,
                           "File with dictionary definitions for dev testing.")
# Test set added by JP. First 400 lines are crossword questions, last 250 are concept descriptions 
# (as described in Hill et al. (2015))
tf.app.flags.DEFINE_string("test_file", config.test_file,
                           "File with dictionary definitions for CROSSWORD testing.")

tf.app.flags.DEFINE_string("save_dir", config.save_dir, "Directory for saving model."
                           "If using restore=True, directory to restore from.")
tf.app.flags.DEFINE_boolean("restore", config.restore, "Restore a trained model"
                            "instead of training one.")
tf.app.flags.DEFINE_boolean("evaluate", config.evaluate, "Evaluate model (needs" 
                            "Restore==True).")
tf.app.flags.DEFINE_string("vocab_file", config.vocab_file, "Path to vocab file")
tf.app.flags.DEFINE_boolean("pretrained_target", config.pretrained_target,
                            "Use pre-trained embeddings for head words.")
tf.app.flags.DEFINE_boolean("pretrained_input", config.pretrained_input,
                            "Use pre-trained embeddings for gloss words.")
tf.app.flags.DEFINE_string("embeddings_path",
                           config.embeddings_path,
                           "Path to pre-trained (.pkl) word embeddings.")

# All experiments were carried out using the recurrent network
tf.app.flags.DEFINE_string("encoder_type", "recurrent", "BOW or recurrent.")
tf.app.flags.DEFINE_string("model_name", "recurrent", "BOW or recurrent.")


FLAGS = tf.app.flags.FLAGS

outfile = config.outfile
with open(outfile, 'w') as f:
    pass # clear the outfile incase it already contains text


def read_data(data_path, vocab_size, phase="train"):
  """Read data from gloss and head files.

  Args:
    data_path: path to the definition .gloss and .head files.
    vocab_size: total number of word types in the data.
    phase: used to locate definitions (train or dev).

  Returns:
    a tuple (gloss, head, incorrect_head)
      where gloss is an np array of encoded glosses and 
      head is an encoded array of head words; len(gloss) == len(head).
      incorrect_head is used for rank loss training only, and is a 
      randomly shuffled version of head array, so that the word != the correct word
      
  """
  glosses, heads, incorrect_heads = [], [], []
  gloss_path = os.path.join(
      data_path, "%s.definitions.ids%s.gloss" % (phase, vocab_size))
  head_path = os.path.join(
      data_path, "%s.definitions.ids%s.head" % (phase, vocab_size))
  with tf.gfile.GFile(gloss_path, mode="r") as gloss_file:
    
    # JP: DYNAMIC PADDING to allow variable length sequences. All batches must be the same size for processing
    # find padding length for each batch
    all_glosses = gloss_file.readlines()
    gloss_lens = [len(gloss.split()) for gloss in all_glosses]
    num_batches = len(gloss_lens) // FLAGS.batch_size
    max_pad = []
    for i in range(num_batches):
      padding_required = [max(gloss_lens[FLAGS.batch_size * i: FLAGS.batch_size * (i+1)])] * FLAGS.batch_size
      if padding_required[0] > FLAGS.max_seq_len:
        padding_required = [FLAGS.max_seq_len]*FLAGS.batch_size
      max_pad.extend(padding_required)
    # add zeroes to sequence to make it correct length, these won't be used later due to batch size
    if len(max_pad) < len(gloss_lens):
      max_pad.extend([0] * (len(gloss_lens) - len(max_pad)))
    
  with tf.gfile.GFile(gloss_path, mode="r") as gloss_file:  
    with tf.gfile.GFile(head_path, mode="r") as head_file:
      gloss, head = gloss_file.readline(), head_file.readline()
      counter = 0
      while gloss and head:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        gloss_ids = np.array([int(x) for x in gloss.split()], dtype=np.int32)
        # Pad each sequence according to longest sequence in each batch
        if phase == 'train':
          gloss_ids = np.array(data_utils_BPE.pad_sequence(gloss_ids, max_pad[counter-1]), dtype=np.int32)
        else:
          pass
        glosses.append(gloss_ids)
        heads.append(int(head))
        incorrect_heads.append(int(head))
        gloss, head = gloss_file.readline(), head_file.readline()

  heads_out = np.array(heads, dtype=np.int32)
  incorrect_heads_out = np.array(incorrect_heads, dtype=np.int32)
  # Collect incorrect target word for rank loss
  np.random.shuffle(incorrect_heads_out)
  # Shuffle still results in some (<1%) correct target words for the gloss, 
  # Apply brute-force method to ensure these are different:
  for i, check in enumerate(heads_out):
    if incorrect_heads[i]==check:
      if check > 101:
        incorrect_heads[i] = check-100
      else:
        incorrect_heads[i] = check+100
  return np.asarray(glosses), heads_out, incorrect_heads_out


def load_pretrained_embeddings(embeddings_file_path):
  """Loads pre-trained word embeddings.

  Args:
    embeddings_file_path: path to the pickle file with the embeddings.

  Returns:
    tuple of (dictionary of embeddings, length of each embedding).
  """
  print("Loading pretrained embeddings from %s" % embeddings_file_path)
  with open(embeddings_file_path, "rb") as input_file:
    pre_embs_dict = pickle.load(input_file, encoding='bytes')
  iter_keys = iter(pre_embs_dict.keys())
  first_key = next(iter_keys)
  embedding_length = len(pre_embs_dict[first_key])
  print("%d embeddings loaded; each embedding is length %d" %
        (len(pre_embs_dict.values()), embedding_length))
  return pre_embs_dict, embedding_length


def get_embedding_matrix(embedding_dict, vocab, emb_dim):
  emb_matrix = np.zeros([len(vocab), emb_dim])
  for word, ii in vocab.items():
    if word in embedding_dict:
      emb_matrix[ii] = embedding_dict[word]
    else:
#      print("OOV word when building embedding matrix: ", word)
      pass
  return np.asarray(emb_matrix)
  


def gen_batch(raw_data, batch_size):
  '''
  This generates a batch of head/gloss pairs.
  Edited by JP to yield incorrect_head as well
  data_x is the batch for gloss, _fw and _bw refers to direction (for bidirectional) 
  data_y is the batch for correct head 
  data_z is the batch for incorrect head
  '''
  raw_x, raw_y, raw_z = raw_data
  data_length = len(raw_y)
  num_batches = data_length // batch_size
  data_x_fw, data_x_bw , data_y, data_z = [], [], [], []
  for i in range(num_batches):
    data_x_fw = raw_x[batch_size * i:batch_size * (i + 1)]
    data_x_bw = [definition[::-1] for definition in data_x_fw]
    data_y = raw_y[batch_size * i:batch_size * (i + 1)]
    data_z = raw_z[batch_size * i:batch_size * (i + 1)]
    yield (data_x_fw, data_x_bw, data_y, data_z)



def gen_epochs(data_path, total_epochs, batch_size, vocab_size, phase="train"):
  # Read all of the glosses and heads into two arrays.
  raw_data = read_data(data_path, vocab_size, phase)
  # Return a generator over the data.
  for _ in range(total_epochs):
    yield gen_batch(raw_data, batch_size)


def build_model(max_seq_len, vocab_size, emb_size, learning_rate, encoder_type,
                pretrained_target=True, pretrained_input=False, pre_embs=None):
  """Build the dictionary model including loss function.

  Args:
    max_seq_len: maximum length of gloss.
    vocab_size: number of words in vocab.
    emb_size: size of the word embeddings.
    learning_rate: learning rate for the optimizer.
    encoder_type: method of encoding (RRN or BOW).
    pretrained_target: Boolean indicating pre-trained head embeddings.
    pretrained_input: Boolean indicating pre-trained gloss word embeddings.
    pre_embs: pre-trained embedding matrix.

  Returns:
    tuple of (gloss_in, head_in, total_loss, train_step, output_form)

  Creates the embedding matrix for the input, which is split into the
  glosses (definitions) and the heads (targets). So checks if there are
  pre-trained embeddings for the glosses or heads, and if not sets up
  some trainable embeddings. The default is to have pre-trained
  embeddings for the heads but not the glosses.

  The encoder for the glosses is either an RNN (with LSTM cell) or a
  bag-of-words model (in which the word vectors are simply
  averaged). For the RNN, the output is the output vector for the
  final state.

  If the heads are pre-trained, the output of the encoder is put thro'
  a non-linear layer, and the loss is the cosine distance. Without
  pre-trained heads, a linear layer on top of the encoder output is
  used to predict logits for the words in the vocabulary, and the loss
  is cross-entropy.
  """
  # Build the TF graph on the GPU.
  with tf.device("/device:GPU:0"):
    tf.reset_default_graph()
    # Batch of input definitions (glosses).
    # create a forward and backward representation of the gloss. Only fw is used in LSTM average
    gloss_in_fw = tf.placeholder(
        tf.int32, [None, None], name="fw_input_placeholder")
    gloss_in_bw = tf.placeholder(
        tf.int32, [None, None], name="bw_input_placeholder")
    
    # Batch of the corresponding targets (heads).
    head_in = tf.placeholder(tf.int32, [None], name="labels_placeholder")
    # Batch of random incorrect heads (for rank loss)
    incorrect_head_in = tf.placeholder(tf.int32, [None], name="incorrect_placeholder")
    with tf.variable_scope("embeddings"):
      if pretrained_input:
        assert pre_embs is not None, "Must include pre-trained embedding matrix"
        # embedding_matrix is pre-trained embeddings.
        embedding_matrix = tf.get_variable(
            name="inp_emb",
            shape=[vocab_size, emb_size],
            initializer=tf.constant_initializer(pre_embs),
            trainable=False)
      else:
        # embedding_matrix is learned.
        embedding_matrix = tf.get_variable(
            name="inp_emb",
            shape=[vocab_size, emb_size])
    # embeddings for the batch of definitions (glosses).
      embs = tf.nn.embedding_lookup(embedding_matrix, gloss_in_fw)
    if pretrained_target:
      out_size = pre_embs.shape[-1]
    else:
      out_size = emb_size
    # RNN encoder for the definitions.
    if encoder_type == "recurrent":
      # both LSTM types added by JP
      if config.LSTM_type=='average':
        cell = tf.nn.rnn_cell.LSTMCell(emb_size)
        # state is the final state of the RNN.
        rnn_output, state = tf.nn.dynamic_rnn(cell, embs, dtype=tf.float32)
        # mean_state is the mean of RNN outputs across the sequence.
        mean_state = tf.reduce_mean(rnn_output, axis=1)
        core_out = mean_state
      elif config.LSTM_type=='bidirectional':
        # create the forward and backward LSTMs
        cell_fw = tf.nn.rnn_cell.LSTMCell(emb_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(emb_size)
        # state is the final state of the RNN.
        rnn_output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embs, dtype=tf.float32)
        # mean_state is the mean of RNN outputs across the sequence.
        final_concat = tf.concat(state, 2)
        core_out = final_concat[0]
      else:
        print('LSTM_type not set in config')
        raise NotImplementedError
    else:
      core_out = tf.reduce_mean(embs, axis=1)
    # core_out is the output from the gloss encoder.
    output_form = "cosine"
    if pretrained_target:
      # Create a loss based on cosine distance for pre-trained heads.
      if pretrained_input:
        # Already have the pre-trained embedding matrix, so use that.
        out_emb_matrix = embedding_matrix
      else:
        # Target embeddings are pre-trained.
        out_emb_matrix = tf.get_variable(
            name="out_emb",
            shape=[vocab_size, out_size],
            initializer=tf.constant_initializer(pre_embs),
            trainable=False)
      # Put core_out thro' a non-linear layer.

      core_out = tf.contrib.layers.fully_connected(
          core_out,
          out_size,
          activation_fn=tf.tanh)

      # Embeddings for the batch of targets/heads.
      targets = tf.nn.embedding_lookup(out_emb_matrix, head_in)
      # Embeddings for random incorrect heads (for rank loss).
      with tf.variable_scope('Loss_Layer'):
        if config.loss=='cosine':
      # cosine_distance assumes the arguments are unit normalized.
          losses = tf.losses.cosine_distance(
              tf.nn.l2_normalize(targets, 1),
              tf.nn.l2_normalize(core_out, 1),
              dim=1)
          
        elif config.loss=='rank':
      # JP: adapted from https://hanxiao.github.io/2017/11/08/Optimizing-Contrastive-Rank-Triplet-Loss-in-Tensorflow-for-Neural/
          incorrect = tf.nn.embedding_lookup(out_emb_matrix, incorrect_head_in)
          weight = 1
          margin = 0.1
          y_norm = tf.nn.l2_normalize(core_out, 1)
          d_pos_norm = tf.nn.l2_normalize(targets, 1)
          d_neg_norm = tf.nn.l2_normalize(incorrect, 1)
          metric_p = tf.reduce_sum(y_norm * d_pos_norm, axis=1, name='cos_sim_pos')
          metric_n = tf.reduce_sum(y_norm * d_neg_norm, axis=1, name='cos_sim_neg')
          delta = metric_n - metric_p
          loss_q_pos = tf.nn.relu(margin + delta)
          losses = tf.reduce_sum(weight * loss_q_pos)
        else:
          print('loss not set in config')
          raise NotImplementedError
    
    else:
      # Create a softmax loss when no pre-trained heads.
      out_emb_matrix = tf.get_variable(
          name="out_emb", shape=[emb_size, vocab_size])
      logits = tf.matmul(core_out, out_emb_matrix)
      pred_dist = tf.nn.softmax(logits, name="predictions")
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=head_in, logits=pred_dist)
      output_form = "softmax"
    # Average loss across batch.
    total_loss = tf.reduce_mean(losses, name="total_loss")

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    return gloss_in_fw, gloss_in_bw, head_in, incorrect_head_in, total_loss, train_step, output_form



def train_network(model, num_epochs, batch_size, data_dir, save_dir,
                  vocab_size, name="model", verbose=True):
  '''
  JP: Training function extended to allow evaluation for each epoch
  '''
  # Running count of the number of training instances.
  num_training = 0
  # saver object for saving the model after each epoch.
  saver = tf.train.Saver()
  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    gloss_in_fw, gloss_in_bw, head_in, incorrect_in, total_loss, train_step, _ = model
    # Initialize the model parameters.
    sess.run(tf.global_variables_initializer())
    # Record all training losses for potential reporting.
    training_losses = []
    # epoch is a generator of batches which passes over the data once.
    for idx, epoch in enumerate(
        gen_epochs(
            data_dir, num_epochs, batch_size, vocab_size, phase="train")):
      # Running total for training loss reset every 500 steps.
      training_loss = 0
      if verbose:
        with open(outfile, 'a') as f:
          print("\nEPOCH", idx, file=f)
          
      for step, (gloss_fw, gloss_bw, head, incorrect) in enumerate(epoch):
      # Glosses come out as a list because not all equal lengths, convert to array
        gloss_batch_fw = np.array([array for array in gloss_fw], dtype=np.int32)
        gloss_batch_bw = np.array([array for array in gloss_bw], dtype=np.int32)
        num_training += len(gloss_fw)
        if config.LSTM_type=='average':
          training_loss_, _ = sess.run(
              [total_loss, train_step],
              feed_dict={gloss_in_fw: gloss_batch_fw, head_in: head, incorrect_in: incorrect})
        elif config.LSTM_type=='bidirectional':
          training_loss_, _ = sess.run(
              [total_loss, train_step],
              feed_dict={gloss_in_fw: gloss_batch_fw, gloss_in_bw: gloss_batch_bw, head_in: head, incorrect_in: incorrect})
        training_loss += training_loss_
        if step % 500 == 0 and step > 0:
          if verbose:
            loss_ = training_loss / 500
            with open(outfile, 'a') as f:
              print("Average loss step %s, for last 500 steps: %s"
                % (step, loss_), file=f)
          training_losses.append(training_loss / 500)
          training_loss = 0          

      # Save current model after another epoch.
      save_path = os.path.join(save_dir, "%s_%s.ckpt" % (name, idx))
      save_path = saver.save(sess, save_path)
      print("Model saved in file: %s after epoch: %s" % (save_path, idx))
      
      # JP: run the evaluation routine (for each epoch)
      if FLAGS.evaluate:
        with tf.device("/cpu:0"):
          with open(outfile, 'a') as f:
            print('evaluating while training', file=f)
          if FLAGS.vocab_file is None:
            vocab_file = os.path.join(FLAGS.data_dir,
                                  "definitions_%s.vocab" % FLAGS.vocab_size)
          else:
            vocab_file = FLAGS.vocab_file
            
          if FLAGS.pretrained_input or FLAGS.pretrained_target:
            embs_dict, pre_emb_dim = load_pretrained_embeddings(FLAGS.embeddings_path)
            vocab, _ = data_utils_BPE.initialize_vocabulary(vocab_file)
            pre_embs = get_embedding_matrix(embs_dict, vocab, pre_emb_dim)
          
          out_form = "cosine"
          graph = tf.get_default_graph()
          # get the names of input and output tensors
          input_node_fw = graph.get_tensor_by_name("fw_input_placeholder:0")
          input_node_bw = graph.get_tensor_by_name("bw_input_placeholder:0")
            
          target_node = graph.get_tensor_by_name("labels_placeholder:0")
          if out_form == "softmax":
            predictions = graph.get_tensor_by_name("predictions:0")
          else:
            predictions = graph.get_tensor_by_name("fully_connected/Tanh:0")
          loss = graph.get_tensor_by_name("total_loss:0") # not used in evaluation

          evaluate_model(sess, FLAGS.data_dir,
                         input_node_fw, input_node_bw, target_node,
                         predictions, loss, embs=pre_embs, out_form="cosine")

          
      # Remove older model versions from previous epochs to minimize HDD usage
      if idx>0:
        os.remove(os.path.join(save_dir, "%s_%s.ckpt.data-00000-of-00001" % (name, idx-1)))
        os.remove(os.path.join(save_dir, "%s_%s.ckpt.index" % (name, idx-1)))
        os.remove(os.path.join(save_dir, "%s_%s.ckpt.meta" % (name, idx-1)))
        with open(outfile, 'a') as f:
          print('deleting old files ', "%s_%s.ckpt" % (name, idx-1), file=f)
        
    print("Total data points seen during training: %s" % num_training)
    return save_dir, saver

def evaluate_model(sess, data_dir, input_node_fw, input_node_bw, target_node, prediction,
                   loss, embs, out_form="cosine"): 
  '''
  Runs the evaluation routine. Added by JP.
  Inputs: 
    data_dir: directory for retrieving test examples
    input_node: input placeholder on graph
    target_node: target placeholder on graph
    prediction: prediction placeholder on graph (output embedding, after tanh layer)
    loss: (present in original implementation but not used here)
    embs: embeddings file 
  
  Uses batch size of 1 to get a prediction (embedding) for each training example, this is 
  compared with all embeddings in the vocabulary using cosine similarity. 
  Note that for crossword questions, only words of correct length are considered. Greatly
  reducing output vocabulary.
  The rank of the correct word is calculated using np.where, and median rank 
  across the test set is reported.
  
  Returns:
    None
  Results:
    If restore = True:
      Saves the results on the various test sets into separate CSV files alongside the correct word.
      Allows for further analysis of results
    If restore = False:
      Evaluating during training, simply outputs the median rank on each test set to the outfile.
      Allows to evaluate model performance during training (to check for overfitting)
  '''
  num_epochs = 1
  batch_size = 1
  check = False
  check_words = []
  print('evaluating model on dev set')
  predictions = np.empty((0,300), dtype=float)
  correct_word = np.empty((0), dtype=int)
  ranks = np.empty((0), dtype=int)
  
  # read the test data using gen_epochs
  for epoch in gen_epochs(
      data_dir, num_epochs, batch_size, FLAGS.vocab_size, phase="test"):
    for (gloss_fw, gloss_bw, head, _) in epoch:
      gloss_batch_fw = np.array([array for array in gloss_fw], dtype=np.int32)
      gloss_batch_bw = np.array([array for array in gloss_bw], dtype=np.int32)
  # use sess.run and feed_dict to get a prediction
      if config.LSTM_type=='average':
        prediction_ = sess.run(
            prediction, feed_dict={input_node_fw: gloss_batch_fw, target_node: head})
      elif config.LSTM_type=='bidirectional':
        prediction_ = sess.run(
            prediction, feed_dict={input_node_fw: gloss_batch_fw, input_node_bw: gloss_batch_bw, target_node: head})
      
      correct_word = np.append(correct_word, head, axis=0)
      predictions = np.append(predictions, prediction_, axis=0)

  sims = 1 - np.squeeze(dist.cdist(predictions, embs, metric="cosine"))
  sims = np.nan_to_num(sims)
  vocab, rev_vocab = data_utils_BPE.initialize_vocabulary(FLAGS.vocab_file)
  # create a list of all the real correct words (not IDs)
  real_word = [rev_vocab[idx] for idx in correct_word]
  # find lengths of these words (for crossword clues)
  real_word_len = [len(word) for word in real_word]

  vocab_list = np.empty((0), dtype=int)
  # pred_array is a list of cosine similarity values for all words in vocab,
  for idx, pred_array in enumerate(sims[:400]):
  # find IDs for all words of correct length:
    for word in vocab:
      if len(word) == real_word_len[idx]:
        vocab_list = np.append(vocab_list, [vocab[word]], axis=0)
    correct_length_ids = vocab_list
    ranked_wids = pred_array.argsort()[::-1]
    words = [word for word in ranked_wids if word in correct_length_ids]
    
    xword_rank = np.where(words==correct_word[idx])
    ranks = np.append(ranks, xword_rank)
    vocab_list = np.empty((0), dtype=int)

  # find rank for definitions (non-crossword clues)
  counter = 400 # cant loop through the idx because it will start at 0 again
  for idx, pred_array in enumerate(sims[400:]):
    rank = np.where(pred_array.argsort()[::-1]==correct_word[counter])
    ranks = np.append(ranks, rank)
    counter+=1
  # find rank for crossword clues
    
    if check: # check undeperforming words to see top candidates
      if idx > 250:
        if rank[0] > 90000:
          check_words.append('HEAD WORD: {}'.format(rev_vocab[correct_word[idx]]))
          for candidate in pred_array.argsort()[::-1][:10]:
            check_words.append(rev_vocab[candidate])
            
  if check:
    print(check_words)

# Test set composition:
# guardian_long[:100], 
# guardian_shor[100:200], 
# NYT_long[200:300], 
# NYT_short[300:400],
# eval_set[400]

  if FLAGS.restore:
    guardian_long = pd.DataFrame({'head WID':correct_word[:100],'rank':ranks[:100]})
    guardian_short = pd.DataFrame({'head WID':correct_word[100:200],'rank':ranks[100:200]})
    NYT_long = pd.DataFrame({'head WID':correct_word[200:300],'rank':ranks[200:300]})
    NYT_short = pd.DataFrame({'head WID':correct_word[300:400],'rank':ranks[300:400]})
    definitions_frame = pd.DataFrame({'head WID':correct_word[400:],'rank':ranks[400:]})
    xword_frame = pd.concat([guardian_long, guardian_short, NYT_long, NYT_short], axis=0, join='inner')  
    xword_frame.to_csv('final_csvs' + FLAGS.save_dir.split('/')[2] + 'x_word.csv', sep=',')
    definitions_frame.to_csv('final_csvs' + FLAGS.save_dir.split('/')[2] + 'definitions.csv', sep=',')
    print('guardian_long median: {}\nguardian_short median: {}\nNYT_long median: {}\nNYT_short median: {}\nDefinitions median: {}'.format(
        np.median(ranks[:100]),np.median(ranks[100:200]),np.median(ranks[200:300]),np.median(ranks[300:400]), np.median(ranks[400:])))
  else:
    with open(outfile, 'a') as f:
      print('guardian_long median: {}\nguardian_short median: {}\nNYT_long median: {}\nNYT_short median: {}\nDefinitions median: {}'.format(
        np.median(ranks[:100]),np.median(ranks[100:200]),np.median(ranks[200:300]),np.median(ranks[300:400]), np.median(ranks[400:])), 
      file=f)
#  print('guardian_long median: {}\nguardian_short median: {}\nNYT_long median: {}\nNYT_short median: {}\nDefinitions median: {}'.format(
#      np.median(ranks[:100]),np.median(ranks[100:200]),np.median(ranks[200:300]),np.median(ranks[300:400]), np.median(ranks[400:])))


def restore_model(sess, save_dir, vocab_file, out_form):
  model_path = tf.train.latest_checkpoint(save_dir)
  # restore the model from the meta graph
  saver = tf.train.import_meta_graph(model_path + ".meta")
  saver.restore(sess, model_path)
  graph = tf.get_default_graph()
  # get the names of input and output tensors
  input_node_fw = graph.get_tensor_by_name("fw_input_placeholder:0")
  input_node_bw = graph.get_tensor_by_name("bw_input_placeholder:0")
  target_node = graph.get_tensor_by_name("labels_placeholder:0")
  if out_form == "softmax":
    predictions = graph.get_tensor_by_name("predictions:0")
  else:
    predictions = graph.get_tensor_by_name("fully_connected/Tanh:0")
  loss = graph.get_tensor_by_name("total_loss:0") # check this is OK
  # vocab is mapping from words to ids, rev_vocab is the reverse.
  vocab, rev_vocab = data_utils_BPE.initialize_vocabulary(vocab_file)
  return input_node_fw, input_node_bw, target_node, predictions, loss, vocab, rev_vocab


# NOTE: The query model function is removed for this implementation. 
#       It will be re-added in a future version


def main(unused_argv):
  """Calls train and test routines for the dictionary model.

  If restore FLAG is true, loads an existing model and runs test
  routine. If restore FLAG is false, builds a model and trains it.
  """
  if FLAGS.vocab_file is None:
    vocab_file = os.path.join(FLAGS.data_dir,
                              "definitions_%s.vocab" % FLAGS.vocab_size)
  else:
    vocab_file = FLAGS.vocab_file

    # Build and train a dictionary model and test every epoch.
  if not FLAGS.restore:
    emb_size = FLAGS.embedding_size
    # Load any pre-trained word embeddings.
    if FLAGS.pretrained_input or FLAGS.pretrained_target:
      # embs_dict is a dictionary from words to vectors.
      embs_dict, pre_emb_dim = load_pretrained_embeddings(FLAGS.embeddings_path)
      if FLAGS.pretrained_input:
        emb_size = pre_emb_dim
    else:
      pre_embs, embs_dict = None, None
  
    # Create vocab file, process definitions (if necessary).
    data_utils_BPE.prepare_dict_data(
        FLAGS.data_dir,
        FLAGS.train_file,
        FLAGS.dev_file,
        FLAGS.test_file,
        vocabulary_size=FLAGS.vocab_size,
        max_seq_len=FLAGS.max_seq_len)
    # vocab is a dictionary from strings to integers.
    vocab, _ = data_utils_BPE.initialize_vocabulary(vocab_file)
    pre_embs = None
    if FLAGS.pretrained_input or FLAGS.pretrained_target:
      # pre_embs is a numpy array with row vectors for words in vocab.
      # for vocab words not in embs_dict, vector is all zeros.
      pre_embs = get_embedding_matrix(embs_dict, vocab, pre_emb_dim)
  
    # Build the TF graph for the dictionary model.
    model = build_model(
        max_seq_len=FLAGS.max_seq_len,
        vocab_size=FLAGS.vocab_size,
        emb_size=emb_size,
        learning_rate=FLAGS.learning_rate,
        encoder_type=FLAGS.encoder_type,
        pretrained_target=FLAGS.pretrained_target,
        pretrained_input=FLAGS.pretrained_input,
        pre_embs=pre_embs)
  
    # Run the training for specified number of epochs.
    save_path, saver = train_network(
        model,
        FLAGS.num_epochs,
        FLAGS.batch_size,
        FLAGS.data_dir,
        FLAGS.save_dir,
        FLAGS.vocab_size,
        name=FLAGS.model_name)
  

  # Load an existing model.
  else:
    # Note cosine loss output form is hard coded here. For softmax output
      # change "cosine" to "softmax"
    if FLAGS.pretrained_input or FLAGS.pretrained_target:
      embs_dict, pre_emb_dim = load_pretrained_embeddings(FLAGS.embeddings_path)
      vocab, _ = data_utils_BPE.initialize_vocabulary(vocab_file)
      pre_embs = get_embedding_matrix(embs_dict, vocab, pre_emb_dim)
      out_form = "cosine"
      print('out form is cosine')
    else:
      out_form = "softmax"
    with tf.device("/cpu:0"):
      with tf.Session() as sess:
        (input_node_fw, input_node_bw, target_node, predictions, loss, vocab,
          rev_vocab) = restore_model(sess, FLAGS.save_dir, vocab_file,
                                     out_form=out_form)
  
        if FLAGS.evaluate:
          evaluate_model(sess, FLAGS.data_dir,
                         input_node_fw, input_node_bw, target_node,
                         predictions, loss, embs=pre_embs, out_form=out_form)
  
        # Load the final saved model and run querying routine.
#        query_model(sess, input_node, predictions,
#                    vocab, rev_vocab, FLAGS.max_seq_len, embs=pre_embs,
#                    out_form="cosine")


if __name__ == "__main__":
  tf.app.run()
