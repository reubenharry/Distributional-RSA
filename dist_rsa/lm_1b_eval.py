# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Eval pre-trained 1 billion word language model.
"""
import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

from google.protobuf import text_format
from dist_rsa.utils.data_utils import CharsVocabulary

FLAGS = tf.flags.FLAGS
# General flags.
tf.flags.DEFINE_string('mode', 'sample',
                       'One of [sample, eval, dump_emb, dump_lstm_emb]. '
                       '"sample" mode samples future word predictions, using '
                       'FLAGS.prefix as prefix (prefix could be left empty). '
                       '"eval" mode calculates perplexity of the '
                       'FLAGS.input_data. '
                       '"dump_emb" mode dumps word and softmax embeddings to '
                       'FLAGS.save_dir. embeddings are dumped in the same '
                       'order as words in vocabulary. All words in vocabulary '
                       'are dumped.'
                       'dump_lstm_emb dumps lstm embeddings of FLAGS.sentence '
                       'to FLAGS.save_dir.')
tf.flags.DEFINE_string('pbtxt', 'dist_rsa/data/graph-2016-09-10.pbtxt',
                       'GraphDef proto text file used to construct model '
                       'structure.')
tf.flags.DEFINE_string('ckpt', 'dist_rsa/data/ckpt-*',
                       'Checkpoint directory used to fill model values.')
tf.flags.DEFINE_string('vocab_file', "dist_rsa/data/vocab-2016-09-10.txt", 'Vocabulary file.')
tf.flags.DEFINE_string('save_dir', '',
                       'Used for "dump_emb" mode to save word embeddings.')
# sample mode flags.
tf.flags.DEFINE_string('prefix', '',
                       'Used for "sample" mode to predict next words.')
tf.flags.DEFINE_integer('max_sample_words', 100,
                        'Sampling stops either when </S> is met or this number '
                        'of steps has passed.')
tf.flags.DEFINE_integer('num_samples', 1,
                        'Number of samples to generate for the prefix.')
# dump_lstm_emb mode flags.
tf.flags.DEFINE_string('sentence', '',
                       'Used as input for "dump_lstm_emb" mode.')
# eval mode flags.
tf.flags.DEFINE_string('input_data', '',
                       'Input data files for eval model.')
tf.flags.DEFINE_integer('max_eval_steps', 1000000,
                        'Maximum mumber of steps to run "eval" mode.')


# For saving demo resources, use batch size 1 and step 1.
BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50


def _LoadModel(gd_file, ckpt_file):
  print(gd_file)
  """Load the model from GraphDef and Checkpoint.

  Args:
    gd_file: GraphDef proto text file.
    ckpt_file: TensorFlow Checkpoint file.

  Returns:
    TensorFlow session and tensors dict.
  """
  with tf.Graph().as_default():
    sys.stderr.write('Recovering graph.\n')
    with tf.gfile.FastGFile(gd_file, 'r') as f:
      # s = f.read().decode()
      s = f.read()
      gd = tf.GraphDef()
      text_format.Merge(s, gd)

    tf.logging.info('Recovering Graph %s', gd_file)
    t = {}
    [t['states_init'], t['lstm/lstm_0/control_dependency'],
     t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
     t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
     t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
     t['all_embs'], t['softmax_weights'], t['global_step']
    ] = tf.import_graph_def(gd, {}, ['states_init',
                                     'lstm/lstm_0/control_dependency:0',
                                     'lstm/lstm_1/control_dependency:0',
                                     'softmax_out:0',
                                     'class_ids_out:0',
                                     'class_weights_out:0',
                                     'log_perplexity_out:0',
                                     'inputs_in:0',
                                     'targets_in:0',
                                     'target_weights_in:0',
                                     'char_inputs_in:0',
                                     'all_embs_out:0',
                                     'Reshape_3:0',
                                     'global_step:0'], name='')

    sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run('save/restore_all', {'save/Const:0': ckpt_file})
    sess.run(t['states_init'])

  return sess, t


def _EvalModel(dataset):
  """Evaluate model perplexity using provided dataset.

  Args:
    dataset: LM1BDataset object.
  """
  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  current_step = t['global_step'].eval(session=sess)
  sys.stderr.write('Loaded step %d.\n' % current_step)

  data_gen = dataset.get_batch(BATCH_SIZE, NUM_TIMESTEPS, forever=False)
  sum_num = 0.0
  sum_den = 0.0
  perplexity = 0.0
  for i, (inputs, char_inputs, _, targets, weights) in enumerate(data_gen):
    input_dict = {t['inputs_in']: inputs,
                  t['targets_in']: targets,
                  t['target_weights_in']: weights}
    if 'char_inputs_in' in t:
      input_dict[t['char_inputs_in']] = char_inputs
    log_perp = sess.run(t['log_perplexity_out'], feed_dict=input_dict)

    if np.isnan(log_perp):
      sys.stderr.error('log_perplexity is Nan.\n')
    else:
      sum_num += log_perp * weights.mean()
      sum_den += weights.mean()
    if sum_den > 0:
      perplexity = np.exp(sum_num / sum_den)

    sys.stderr.write('Eval Step: %d, Average Perplexity: %f.\n' %
                     (i, perplexity))

    if i > FLAGS.max_eval_steps:
      break


def _SampleSoftmax(softmax):
  return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)


def _SampleModel(prefix_words, vocab):

  """Predict next words using the given prefix words.

  Args:
    prefix_words: Prefix words.
    vocab: Vocabulary. Contains max word chard id length and converts between
        words and ids.
  """
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  if prefix_words.find('<S>') != 0:
    prefix_words = '<S> ' + prefix_words

  prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
  prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]
  for _ in xrange(FLAGS.num_samples):
    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros(
        [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
    samples = prefix[:]
    char_ids_samples = prefix_char_ids[:]
    sent = ''
    counter = 0
    while True:
      inputs[0, 0] = samples[0]
      char_ids_inputs[0, 0, :] = char_ids_samples[0]
      samples = samples[1:]
      char_ids_samples = char_ids_samples[1:]

      softmax = sess.run(t['softmax_out'],
                         feed_dict={t['char_inputs_in']: char_ids_inputs,
                                    t['inputs_in']: inputs,
                                    t['targets_in']: targets,
                                    t['target_weights_in']: weights})

      # print([vocab.id_to_word(x) for x in range(len(softmax[0]))])
      sorted_clipped_dist = np.argsort(-softmax[0])[:100]
      # print([(vocab.id_to_word(x),x) for x in sorted_clipped_dist])

      sample = _SampleSoftmax(softmax[0])
      print(sample)
      # raise Exception
      prob_dict = {}
      for i,x in enumerate(softmax[0]):
        prob_dict[vocab.id_to_word(i)] = x

      # print(prob_dict['shark'])
      # print(prob_dict['pig'])
      # animals = list(set(['human','ant','bat','bear','bee','bird','buffalo','cat','cow','dog','dolphin','duck','elephant','fish','fox','frog','goat','goose','horse','kangaroo','lion','monkey','owl','ox','penguin','pig','rabbit','shark','sheep','tiger','whale','wolf','zebra','human','puppy']))
      # from utils.load_data import get_words
      # _,nouns,_,_ = get_words(preprocess=False)
      # nouns = [x for x in nouns if x in prob_dict]
      # sorted_nouns = sorted(nouns,key=lambda x : prob_dict[x],reverse=True)
      # print(sorted_nouns[:20])

      # sample_char_ids = vocab.word_to_char_ids(vocab.id_to_word(sample))
      # if not samples:
      #   samples = [sample]
      #   char_ids_samples = [sample_char_ids]
      # sent += vocab.id_to_word(samples[0]) + ' '
      # print(sent)
      # # # print(len(FLAGS.prefix.split(" ")),FLAGS.prefix.split(" "))
      # if counter > len(FLAGS.prefix.split(" "))+2:
      # #   # return prob_dict
      # #   return prob_dict
      #   break

      return prob_dict
      # counter += 1



      # if (vocab.id_to_word(samples[0]) == '</S>' or
      #     len(sent) > FLAGS.max_sample_words):
      #   break


def _DumpEmb(vocab):
  """Dump the softmax weights and word embeddings to files.

  Args:
    vocab: Vocabulary. Contains vocabulary size and converts word to ids.
  """
  assert FLAGS.save_dir, 'Must specify FLAGS.save_dir for dump_emb.'
  inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  softmax_weights = sess.run(t['softmax_weights'])
  fname = FLAGS.save_dir + '/embeddings_softmax.npy'
  with tf.gfile.Open(fname, mode='w') as f:
    np.save(f, softmax_weights)
  sys.stderr.write('Finished softmax weights\n')

  all_embs = np.zeros([vocab.size, 1024])
  for i in xrange(vocab.size):
    input_dict = {t['inputs_in']: inputs,
                  t['targets_in']: targets,
                  t['target_weights_in']: weights}
    if 'char_inputs_in' in t:
      input_dict[t['char_inputs_in']] = (
          vocab.word_char_ids[i].reshape([-1, 1, MAX_WORD_LEN]))
    embs = sess.run(t['all_embs'], input_dict)
    all_embs[i, :] = embs
    sys.stderr.write('Finished word embedding %d/%d\n' % (i, vocab.size))

  fname = FLAGS.save_dir + '/embeddings_char_cnn.npy'
  with tf.gfile.Open(fname, mode='w') as f:
    np.save(f, all_embs)
  sys.stderr.write('Embedding file saved\n')


def _DumpSentenceEmbedding(sentence, vocab):
  """Predict next words using the given prefix words.

  Args:
    sentence: Sentence words.
    vocab: Vocabulary. Contains max word chard id length and converts between
        words and ids.
  """
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  if sentence.find('<S>') != 0:
    sentence = '<S> ' + sentence

  word_ids = [vocab.word_to_id(w) for w in sentence.split()]
  char_ids = [vocab.word_to_char_ids(w) for w in sentence.split()]

  inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  char_ids_inputs = np.zeros(
      [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
  for i in xrange(len(word_ids)):
    inputs[0, 0] = word_ids[i]
    char_ids_inputs[0, 0, :] = char_ids[i]

    # Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
    # LSTM.
    lstm_emb = sess.run(t['lstm/lstm_1/control_dependency'],
                        feed_dict={t['char_inputs_in']: char_ids_inputs,
                                   t['inputs_in']: inputs,
                                   t['targets_in']: targets,
                                   t['target_weights_in']: weights})

    fname = os.path.join(FLAGS.save_dir, 'lstm_emb_step_%d.npy' % i)
    with tf.gfile.Open(fname, mode='w') as f:
      np.save(f, lstm_emb)
    sys.stderr.write('LSTM embedding step %d file saved\n' % i)


def predict(input_prefix="the man is a"):

  vocab = CharsVocabulary(FLAGS.vocab_file, MAX_WORD_LEN)
  prob_dict = (_SampleModel(input_prefix, vocab))

  # from utils.load_data import get_words
  # _,nouns,_,_ = get_words(preprocess=False)
  # nouns = [x for x in nouns if x in prob_dict]
  # sorted_nouns = sorted(nouns,key=lambda x : prob_dict[x],reverse=True)
  # print(sorted_nouns[:20])
  return prob_dict

def main(input_prefix="the man is a"):

  vocab = CharsVocabulary(FLAGS.vocab_file, MAX_WORD_LEN)
  prob_dict = (_SampleModel("the man is a", vocab))
  print(prob_dict[list(prob_dict)[0]])

  # from utils.load_data import get_words
  # _,nouns,_,_ = get_words(preprocess=False)
  # nouns = [x for x in nouns if x in prob_dict]
  # sorted_nouns = sorted(nouns,key=lambda x : prob_dict[x],reverse=True)
  # print(sorted_nouns[:20])

  # if FLAGS.mode == 'eval':
  #   dataset = data_utils.LM1BDataset(FLAGS.input_data, vocab)
  #   _EvalModel(dataset)
  # elif FLAGS.mode == 'sample':
  #   _SampleModel(FLAGS.prefix, vocab)
  # elif FLAGS.mode == 'dump_emb':
  #   _DumpEmb(vocab)
  # elif FLAGS.mode == 'dump_lstm_emb':
  #   _DumpSentenceEmbedding(FLAGS.sentence, vocab)
  # else:
  #   raise Exception('Mode not supported.')


if __name__ == '__main__':
  tf.app.run()
