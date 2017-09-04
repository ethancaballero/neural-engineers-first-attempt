#!/usr/bin/env python
# -*- coding: utf-8 -*-

# altered from github.com/IndicoDataSolutions/Passage/blob/master/passage/preprocessing.py#L37

import string

puncs_and_digits = set(string.punctuation+string.digits)
puncs_and_digits.remove('_')
puncs_and_digits.add('\n')
puncs_and_digits.add('\t')
puncs_and_digits.add(u'’')
puncs_and_digits.add(u'‘')
puncs_and_digits.add(u'“')
puncs_and_digits.add(u'”')
puncs_and_digits.add(u'´')
puncs_and_digits.add('')

def tokenize_english_text(text):
    string_mode = False
    single_quote_mode = False
    double_quote_mode = False    
    tokenized = []
    w = ''
    for t in text:
        #single_quote_toggle
        if t == "'":
            if double_quote_mode == False:
                    string_mode = not string_mode
                    single_quote_mode = not single_quote_mode
            else:
                pass
        #double_quote_toggle
        if t == '"':
            if single_quote_mode == False:
                string_mode = not string_mode
                double_quote_mode = not double_quote_mode
            else:
                pass
        if string_mode == True:
            tokenized.append(w)
            tokenized.append(t)
        elif t in puncs_and_digits:
            tokenized.append(w)
            tokenized.append(t)
            w = ''
        elif t == ' ':
            tokenized.append(w)
            #tokenized.append(t)
            w = ''
        else:
            w += t
    if w != '':
        tokenized.append(w)
        tokenize
    tokenized = [token for token in tokenized if token]
    return tokenized

a = '''Description: ¶ Name string is a string consisting of letters "R","K" and "V". Today Oz wants to design a name string in a beautiful manner. Actually Oz cannot insert these three letters arbitrary anywhere ,he has to follow some rules to make the name string look beautiful. First thing is that the name string should consist of at most two different letters. Secondly adjacent letters in name string must be different. ¶ ¶ After this procedure Oz wants name string to be as long as possible. Given the number of "R","K" and "V" letters that you have initially ,help Oz to find the maximum length of name string that Oz can make. ¶ ¶ Input : ¶ The first line contains the number of test cases T . Each test case consists of three space separated integers - A,B and C representing number of "R" letters, number of "K" letters and number of "V" letters respectively.  ¶ ¶ Output : ¶ For each test case T, output maximum length of name string that Oz can make. ¶ ¶ Constraints : ¶ 1 ≤ T ≤100 ¶ 0 ≤ A,B,C ≤10^6 ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 1 2 5 ¶ 0 0 2 ¶ ¶ SAMPLE OUTPUT ¶ 5 ¶ 1 ¶ ¶ Explanation ¶ ¶ For first sample : ¶ The longest name string possible is :  VKVKV  using 3 "V" letters and 2 "K" letters and its length is 5.'''

'''
b = tokenize_english_text(a)
print len(b)
print b
'''

'''
punc = string.punctuation
#s = 'bla. bla? bla.bla! bla...'

import re, string

my_string = 'This . is , a string ? with ! punctuation'

regex = re.compile('[%s]' % re.escape(string.punctuation))

print(regex.sub('\1', my_string))
'''

"""
import os
#from spacy.en import English, LOCAL_DATA_DIR
import spacy
en_nlp = spacy.load('en')
#print 1
#question = '''Rani works at sweet shop and is currently arranging sweets i.e. putting sweets in boxes . There are some boxes which contains sweets initially but they are not full . Rani have to fill boxes completely thereby minimizing the total no. of boxes . ¶ You are given N no. of boxes, their initial amount and total capacity in grams . You have to print out the minimum no. of boxes required to store given amount of sweets . ¶ ¶ Input: ¶ First line contains no. of test cases . Each test case contains N no. of boxes and next two lines contains  n values each denoting their initial amount and total capacity . ¶ Output: ¶ Print out the minimum no. of boxes required to store given amount of sweets . ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 3 ¶ 300 525 110 ¶ 350 600 115 ¶ 6 ¶ 1 200 200 199 200 200 ¶ 1000 200 200 200 200 200 ¶ ¶ SAMPLE OUTPUT ¶ 2 ¶ 1 ¶ ¶ Explanation ¶ ¶ Testcase 1: ¶ One way to pack the sweets onto as few boxes as possible is as follows . First, move 50 g from box 3 to box 1, completely filling it up . Next, move the remaining 60 g from box 3 to box 2 . There are now two boxes which contain sweets after this process, so your output should be 2 . ¶ Testcase 2: ¶ One way to consolidate the sweets would be to move the 1 g from box 1 to box 4 . However, this is a poor choice, as it results in only one free box and five boxes which still contain sweets . A better decision is to move all the sweets from the other five boxes onto box 1 . Now there is only one box which contains sweets . Since this is the optimal strategy, the output should be 1 . '''
question = '''Rani works at sweet shop and is currently arranging sweets i.e. putting sweets in boxes. There are some boxes which contains sweets initially but they are not full. Rani have to fill boxes completely thereby minimizing the total no. of boxes. ¶ You are given N no. of boxes, their initial amount and total capacity in grams. You have to print out the minimum no. of boxes required to store given amount of sweets. ¶ ¶ Input: ¶ First line contains no. of test cases. Each test case contains N no. of boxes and next two lines contains  n values each denoting their initial amount and total capacity. ¶ Output: ¶ Print out the minimum no. of boxes required to store given amount of sweets. ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 3 ¶ 300 525 110 ¶ 350 600 115 ¶ 6 ¶ 1 200 200 199 200 200 ¶ 1000 200 200 200 200 200 ¶ ¶ SAMPLE OUTPUT ¶ 2 ¶ 1 ¶ ¶ Explanation ¶ ¶ Testcase 1: ¶ One way to pack the sweets onto as few boxes as possible is as follows. First, move 50 g from box 3 to box 1, completely filling it up. Next, move the remaining 60 g from box 3 to box 2. There are now two boxes which contain sweets after this process, so your output should be 2. ¶ Testcase 2: ¶ One way to consolidate the sweets would be to move the 1 g from box 1 to box 4. However, this is a poor choice, as it results in only one free box and five boxes which still contain sweets. A better decision is to move all the sweets from the other five boxes onto box 1. Now there is only one box which contains sweets. Since this is the optimal strategy, the output should be 1.'''
en_doc = en_nlp(question.decode('utf-8'))
#print 2
#a = sent_detector.tokenize(question.decode('utf-8'))
h = []
for i in en_doc.sents:
    h.append(i)
print len(h)
print h
#'''
for i in h:
    print i
    #'''
    #"""

"""
import nltk
question = '''Rani works at sweet shop and is currently arranging sweets i.e. putting sweets in boxes. There are some boxes which contains sweets initially but they are not full. Rani have to fill boxes completely thereby minimizing the total no. of boxes. ¶ You are given N no. of boxes, their initial amount and total capacity in grams. You have to print out the minimum no. of boxes required to store given amount of sweets. ¶ ¶ Input: ¶ First line contains no. of test cases. Each test case contains N no. of boxes and next two lines contains  n values each denoting their initial amount and total capacity. ¶ Output: ¶ Print out the minimum no. of boxes required to store given amount of sweets. ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 3 ¶ 300 525 110 ¶ 350 600 115 ¶ 6 ¶ 1 200 200 199 200 200 ¶ 1000 200 200 200 200 200 ¶ ¶ SAMPLE OUTPUT ¶ 2 ¶ 1 ¶ ¶ Explanation ¶ ¶ Testcase 1: ¶ One way to pack the sweets onto as few boxes as possible is as follows. First, move 50 g from box 3 to box 1, completely filling it up. Next, move the remaining 60 g from box 3 to box 2. There are now two boxes which contain sweets after this process, so your output should be 2. ¶ Testcase 2: ¶ One way to consolidate the sweets would be to move the 1 g from box 1 to box 4. However, this is a poor choice, as it results in only one free box and five boxes which still contain sweets. A better decision is to move all the sweets from the other five boxes onto box 1. Now there is only one box which contains sweets. Since this is the optimal strategy, the output should be 1.'''
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
a = sent_detector.tokenize(question.decode('utf-8'))
print len(a)
print a
#"""

import segtok.segmenter
import segtok.tokenizer

question = '''Rani works at sweet shop and is currently arranging sweets i.e. putting sweets in boxes. There are some boxes which contains sweets initially...but they are not full. Rani have to fill boxes completely thereby minimizing the total no. of boxes. ¶ You are given N no. of boxes, their initial amount and total capacity in grams. You have to print out the minimum no. of boxes required to store given amount of sweets. ¶ ¶ Input: ¶ First line contains no. of test cases. Each test case contains N no. of boxes and next two lines contains  n values each denoting their initial amount and total capacity. ¶ Output: ¶ Print out the minimum no. of boxes required to store given amount of sweets. ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 3 ¶ 300 525 110 ¶ 350 600 115 ¶ 6 ¶ 1 200 200 199 200 200 ¶ 1000 200 200 200 200 200 ¶ ¶ SAMPLE OUTPUT ¶ 2 ¶ 1 ¶ ¶ Explanation ¶ ¶ Testcase 1: ¶ One way to pack the sweets onto as few boxes as possible is as follows. First, move 50 g from box 3 to box 1, completely filling it up. Next, move the remaining 60 g from box 3 to box 2. There are now two boxes which contain sweets after this process, so your output should be 2. ¶ Testcase 2: ¶ One way to consolidate the sweets would be to move the 1 g from box 1 to box 4. However, this is a poor choice, as it results in only one free box and five boxes which still contain sweets. A better decision is to move all the sweets from the other five boxes onto box 1. Now there is only one box which contains sweets. Since this is the optimal strategy, the output should be 1.'''

s = segtok.segmenter.split_multi(question.decode('utf-8'))

s = segtok.tokenizer.word_tokenizer(question.decode('utf-8'))

#s = segtok.tokenizer.symbol_tokenizer(question.decode('utf-8'))

#'''
for i in s:
    print i
    #'''







#s = re.sub('([.,!?()])', r' \1 ', s)
#s = re.sub('([!"#$%&\\'()*+,-./:;<=>?@[\\]^_`{|}~])', r' \1 ', s)
#s = re.sub('\s{2,}', ' ', s)
#print s