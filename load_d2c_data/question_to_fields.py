# -*- coding: utf-8 -*-

import tensorflow as tf









source_path = '/Users/ethancaballero/Neural-Engineer_Candidates/dmn-tf-alter_working_decoder_d2c/load_d2c_data/train.questions.tok'
target_path = '/Users/ethancaballero/Neural-Engineer_Candidates/dmn-tf-alter_working_decoder_d2c/load_d2c_data/train.answers.tok'

def convert_to_fields(question):
  b = []

  a = question[:-2].replace('¶ ¶ Examples ¶ ', '¦¶ ¶ Examples ¶ ¶ ').replace('¶ ¶ Note ¶ ', '¦¶ ¶ Note ¶ ').split('¦')

  a[0] = '¶ ¶ Description ¶ ' + a[0]
  a = a[0].replace('¶ ¶ Input ¶ ', '¦¶ ¶ Input ¶ ').replace('¶ ¶ Output ¶ ', '¦¶ ¶ Output ¶ ').split('¦') + a[1:]

  for idx, i in enumerate(a):
    if idx == 0 or idx == 1 or idx == 2:
      b.append(i.replace('¶ ¶ Description ¶ ', 'Description¦').replace('¶ ¶ Input ¶ ', 'Input¦').replace('¶ ¶ Output ¶ ', 'Output¦').replace(' . ', ' .¦').split('¦'))
    else:
      b.append(i.replace('¶ ¶ Input ¶ ', '¦¶ ¶ Input ¶ ').replace('¶ ¶ Examples ', 'Examples').replace('¶ ¶ Output ¶ ', '¦¶ Output ¶ ').replace('¶ ¶ Note ¶ ', 'Note¦¶ ').replace(' . ', ' .¦').split('¦'))
      
  for i in range(len(b)):
    b[i] = filter(None, b[i])

  '''
  for i in b:
    print '\n'
    for j in i:
      print j
      #'''

  '''
  for i in b:
    print i[0]
    '''

  return b

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
  #data_set = [[] for _ in _buckets]
  sources = []
  targets = []
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        #print source 
        #print target
        #sources.append(source)
        #targets.append(target)
        sources.append(convert_to_fields(source))
        targets.append(target)
        source, target = source_file.readline(), target_file.readline()
  return sources, targets

sources, targets = read_data(source_path, target_path)

#print len(sources)
#print len(targets)

#print sources
#print targets

question = '''Once when Gerald studied in the first year at school , his teacher gave the class the following homework . She offered the students a string consisting of n small Latin letters ; the task was to learn the way the letters that the string contains are written . However , as Gerald is too lazy , he has no desire whatsoever to learn those letters . That &apos;s why he decided to lose some part of the string ( not necessarily a connected part ) . The lost part can consist of any number of segments of any length , at any distance from each other . However , Gerald knows that if he loses more than k characters , it will be very suspicious . ¶ Find the least number of distinct characters that can remain in the string after no more than k characters are deleted . You also have to find any possible way to delete the characters . ¶ ¶ Input ¶ The first input data line contains a string whose length is equal to n ( 1 ≤ n ≤ 10 ^ 5 ) . The string consists of lowercase Latin letters . The second line contains the number k ( 0 ≤ k ≤ 10 ^ 5 ) . ¶ ¶ Output ¶ Print on the first line the only number m — the least possible number of different characters that could remain in the given string after it loses no more than k characters . ¶ Print on the second line the string that Gerald can get after some characters are lost . The string should have exactly m distinct characters . The final string should be the subsequence of the initial string . If Gerald can get several different strings with exactly m distinct characters , print any of them . ¶ ¶ Examples ¶ Input ¶ aaaaa ¶ 4 ¶ ¶ Output ¶ 1 ¶ aaaaa ¶ ¶ Input ¶ abacaba ¶ 4 ¶ ¶ Output ¶ 1 ¶ aaaa ¶ ¶ Input ¶ abcdefgh ¶ 10 ¶ ¶ Output ¶ 0 ¶ ¶ ¶ ¶ Note ¶ In the first sample the string consists of five identical letters but you are only allowed to delete 4 of them so that there was at least one letter left . Thus , the right answer is 1 and any string consisting of characters &quot; a &quot; from 1 to 5 in length . ¶ In the second sample you are allowed to delete 4 characters . You cannot delete all the characters , because the string has length equal to 7 . However , you can delete all characters apart from &quot; a &quot; ( as they are no more than four ) , which will result in the &quot; aaaa &quot; string . ¶ In the third sample you are given a line whose length is equal to 8 , and k = 10 , so that the whole line can be deleted . The correct answer is 0 and an empty string . ¿'''
