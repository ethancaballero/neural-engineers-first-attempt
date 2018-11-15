
"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "train")
  return train_path


def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "validate"
  dev_path = os.path.join(directory, dev_name)
  return dev_path


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary_source(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)

        for fact in facts:
          for w in fact:
            word = w.encode('UTF-8')
            if word in vocab:
              vocab[word] += 1
            else:
              vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")

def create_vocabulary_target(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)

        tokens=eval(line)
        for w in tokens:
          word = w.replace('\n', '\\n')
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line[:-1] for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  v = [vocabulary.get(w.encode('UTF-8'), UNK_ID) for w in sentence]
  return v


def data_to_token_ids_source(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          fvs=[] 
          for fv in eval(line):
            token_ids = sentence_to_token_ids(fv, vocab, tokenizer,
                                              normalize_digits)
            fvs.append(token_ids)
          tokens_file.write(str(fvs) + "\n")

def data_to_token_ids_target(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          line = line.replace('\\n', '\\\\n')
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(eval(line), vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(str(token_ids) + "\n")

def prepare_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """

  # Get wmt data to the specified directory.
  train_path = get_wmt_enfr_train_set(data_dir)
  dev_path = get_wmt_enfr_dev_set(data_dir)

  # Create vocabularies of the appropriate sizes.
  fr_vocab_path = os.path.join(data_dir, "vocab%d.answers" % fr_vocabulary_size)
  en_vocab_path = os.path.join(data_dir, "vocab%d.questions" % en_vocabulary_size)
  create_vocabulary_source(en_vocab_path, train_path + ".questions", en_vocabulary_size, tokenizer)
  create_vocabulary_target(fr_vocab_path, train_path + ".answers", fr_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  fr_train_ids_path = train_path + (".ids%d.answers" % fr_vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.questions" % en_vocabulary_size)
  data_to_token_ids_target(train_path + ".answers", fr_train_ids_path, fr_vocab_path, tokenizer)
  data_to_token_ids_source(train_path + ".questions", en_train_ids_path, en_vocab_path, tokenizer)

  return (en_train_ids_path, fr_train_ids_path,
        en_train_ids_path, fr_train_ids_path,
        en_vocab_path, fr_vocab_path)

def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens

def get_sentence_lens(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)
        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))
    return lens, sen_lens, max(max_sen_lens)

def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=PAD_ID) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=PAD_ID) for j, s in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len:
                padded_sentences = padded_sentences[(len(padded_sentences)-max_len):]
                lens[i] = max_len
            padded_sentences = np.vstack(padded_sentences)
            padded_sentences = np.pad(padded_sentences, ((0, max_len - lens[i]),(0,0)), 'constant', constant_values=PAD_ID)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]), 'constant', constant_values=PAD_ID) for i, inp in enumerate(inputs)]
    return np.vstack(padded)

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  sources = []
  targets = []

  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      count=0
      while source and target:
        count+=1
        #print (count)
        sources.append(np.array(eval(source)))
        targets.append(np.array([GO_ID]+eval(target)+[EOS_ID]))
        source, target = source_file.readline(), target_file.readline()

  return sources, targets

def pad_length_bucket(source, targets, config, split_sentences=True):
    inputs = source

    if split_sentences:
        input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
        max_mask_len = max_sen_len
    else:
        input_lens = get_lens(inputs)

    t_lens = get_lens(targets)

    max_t_len = np.max(t_lens)
    max_input_len = min(np.max(input_lens), config.max_allowed_inputs)


    #pad out arrays to max
    if split_sentences:
        inputs = pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    targets = pad_inputs(targets, t_lens, max_t_len, "mask")

    if config.train_mode:

        train = targets[:config.num_train], inputs[:config.num_train], t_lens[:config.num_train], input_lens[:config.num_train], input_masks[:config.num_train]
        valid = targets[config.num_train:], inputs[config.num_train:], t_lens[config.num_train:], input_lens[config.num_train:], input_masks[config.num_train:] 

        return train, valid, max_t_len, max_input_len, max_mask_len


    else:
        test = targets, inputs, t_lens, input_lens, input_masks, answers, rel_labels
        return test, max_t_len, max_input_len, max_mask_len

def get_vocab_size(vocab_path):

  with tf.gfile.GFile(vocab_path, mode="r") as vocab_file:
    vocab_line = vocab_file.readline()
    count=0
    while vocab_line:
      count+=1
      vocab_line = vocab_file.readline()

    print(count)

  return count


