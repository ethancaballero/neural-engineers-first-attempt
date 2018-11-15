# -*- coding: utf-8 -*-

import string

import segtok.segmenter
import segtok.tokenizer

from segtok.tokenizer import symbol_tokenizer, word_tokenizer, web_tokenizer
from segtok.tokenizer import split_possessive_markers, split_contractions

'''ADD SECTION FOR CONVERTING EVERYTHING TO LOWER CASE'''

q = '''Description: ¶ Name string is a string consisting of letters "R","K" and "V". Today Oz wants to design a name string in a beautiful manner. Actually Oz's cannot insert these three letters arbitrary anywhere ,he has to follow some rules to make the name string look beautiful. First thing is that the name string should consist of at most two different letters. Secondly adjacent letters in name string must be different. ¶ ¶ After this procedure Oz wants name string to be as long as possible. Given the number of "R","K" and "V" letters that you have initially ,help Oz to find the maximum length of name string that Oz can make. ¶ ¶ Input : ¶ The first line contains the number of test cases T. Each test case consists of three space separated integers - A,B and C representing number of "R" letters, number of "K" letters and number of "V" letters respectively.  ¶ ¶ Output : ¶ For each test case T, output maximum length of name string that Oz can make. ¶ ¶ Constraints : ¶ 1 ≤ T ≤100 ¶ 0 ≤ A,B,C ≤10^6 ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 1 2 asfas 5 ¶ 0 0 2 ¶ ¶ SAMPLE OUTPUT ¶ 5 ¶ 1 ¶ ¶ Explanation ¶ ¶ For first sample : ¶ The longest name string possible is :  VKVKV  using 3 "V" letters and 2 "K" letters and its length is 5.'''

def char_split_if_io_example(sentence):
	'''ADD SECTION FOR CONVERTING EVERYTHING TO LOWER CASE'''
	"""split text into characters"""
	"""used for input/output examples for which char level info is relevant"""
	i = 'Input ¶'
	o = 'Output ¶'

	'''
	i = 'Input \xb6'
	o = 'Output \xb6'
	'''

	sentence_encoded=sentence
	sentence=sentence.decode('utf-8')

	if i in sentence_encoded:
		sentence=sentence_encoded.split(i)
		sentence = word_tokenizer(i.decode('utf-8')) + list(sentence[1])
		s = sentence
		for jdx, j in enumerate(sentence):
			if j == '\xc2':
				s[jdx:jdx+2]=[u'\xb6']
		sentence = s
	elif o in sentence_encoded:
		sentence=sentence_encoded.split(o)
		sentence = word_tokenizer(o.decode('utf-8')) + list(sentence[1])
		s = sentence
		for jdx, j in enumerate(sentence):
			if j == '\xc2':
				s[jdx:jdx+2]=[u'\xb6']
		sentence = s
	else:
		sentence = word_tokenizer(sentence)
	return sentence

def split_nums(list_of_tokens):
	digits = string.digits
	new_list_of_tokens=[]
	for idx, i in enumerate(list_of_tokens):
		digit_present = False
		for j in i:
			if j in digits:
				new_list_of_tokens+=list(i)
				digit_present = True
				break
		if digit_present == False:
			new_list_of_tokens+=[i]

	return new_list_of_tokens

def question_to_tokenized_fields(question):
	b = ['¡ Description']
	a = question.replace('¶ ¶ Examples ¶ ', '¦¶ ¶ Examples ¶ ¶ ').replace('¶ Examples ¶ ', '¦¶ ¶ Examples ¶ ¶ ').split('¦') 

	#You replace Note with Explanation in Codeforces

	#codeforces
	if len(a) > 1:
		for idx, i in enumerate(a):
			if idx == 0:
				c=[]
				c+=[i.encode('utf-8') for i in segtok.segmenter.split_multi(a[idx].decode('utf-8'))]
				for i in c:
					b+=i.replace('¶ ¶ Description ¶ ', '¡ Description¦').replace('¶ ¶ Input ¶ ', '¦¡ Input¦').replace('¶ ¶ Output ¶ ', '¦¡ Output¦').replace('¶ Input ¶ ', '¦¡ Input¦').replace('¶ Output ¶ ', '¦¡ Output¦').replace(' . ', ' .¦').replace('¶ ¶ ', '¦').split('¦')
			else:
				c=[]
				c+=[i.encode('utf-8') for i in segtok.segmenter.split_multi(a[idx].decode('utf-8'))]
				for i in c:
					b+=i.replace('¶ ¶ Input ¶ ', '¦¶ ¶ Input ¶ ').replace('¶ ¶ Examples ', '¡ Examples').replace('¶ Examples ', '¡ Examples').replace('¶ ¶ Output ¶ ', '¦¶ Output ¶ ').replace('¶ ¶ Note ¶ ', '¦¡ Explanation¦').replace('¶ ¶ Input : ¶', '¦¡ Input¦').replace('¶ ¶ Output : ¶', '¦¡ Output¦').replace(' . ', ' .¦').replace('¶ ¶ ', '¦').replace('¶ Output ¶', 'Output ¶').split('¦')

	#hackerearth
	else:
		c=[]
		c+=[i.encode('utf-8') for i in segtok.segmenter.split_multi(a[0].decode('utf-8'))]
		for i in c:
			b+=i.replace('Description: ¶ ', '').replace('¶ ¶ Output', '¶ Output').replace('¶ Output', '¶ ¶ Output').replace('¶ ¶ Input : ¶ ', '¦¡ Input¦').replace('¶ ¶ Output : ¶ ', '¦¡ Output¦').replace('¶ ¶ Input: ¶ ', '¦¡ Input¦').replace('¶ ¶ Output: ¶ ', '¦¡ Output¦').replace('¶ ¶ Input ¶ ', '¦¡ Input¦').replace('¶ ¶ Output ¶ ', '¦¡ Output¦').replace('¶ ¶ Input ', '¦¡ Input¦').replace('¶ ¶ Examples ', '¡ Examples').replace('¶ ¶ Output ', '¦¡ Output¦').replace('¶ ¶ Note ¶ ', '¦¡ Note¦') \
			.replace('¶ ¶ SAMPLE INPUT ¶', '¦¡ Examples¦¶ ¶ Input ¶').replace('¶ ¶ SAMPLE OUTPUT ¶', '¦¶ ¶ Output ¶').replace('¶ ¶ Constraints : ¶ ', '¦¡ Constraints¦').replace('¶ ¶ Constraint : ¶ ', '¦¡ Constraints¦').replace('¶ ¶ Constraints: ¶ ', '¦¡ Constraints¦').replace('¶ ¶ Constraint: ¶ ', '¦¡ Constraints¦').replace('¶ ¶ Constraints ¶ ', '¦¡ Constraints¦').replace('¶ ¶ Constraint ¶ ', '¦¡ Constraints¦').replace('¶ ¶ Explanation ¶ ', '¦¡ Explanation¦').replace('¶ ¶ ', '¦').split('¦')

	b=[split_nums(split_contractions(char_split_if_io_example(x))) for x in b if x.strip()]
	return b

'''ADD SECTION FOR CONVERTING EVERYTHING TO LOWER CASE'''

if __name__ == '__main__':

	q = '''Overall there are m actors in Berland. Each actor has a personal identifier — an integer from 1 to m (distinct actors have distinct identifiers). Vasya likes to watch Berland movies with Berland actors, and he has k favorite actors. He watched the movie trailers for the next month and wrote the following information for every movie: the movie title, the number of actors who starred in it, and the identifiers of these actors. Besides, he managed to copy the movie titles and how many actors starred there, but he didn't manage to write down the identifiers of some actors. Vasya looks at his records and wonders which movies may be his favourite, and which ones may not be. Once Vasya learns the exact cast of all movies, his favorite movies will be determined as follows: a movie becomes favorite movie, if no other movie from Vasya's list has more favorite actors.
	Help the boy to determine the following for each movie:
	 whether it surely will be his favourite movie; whether it surely won't be his favourite movie;  can either be favourite or not.
	Input
	The first line of the input contains two integers m and k (1 ≤ m ≤ 100, 1 ≤ k ≤ m) — the number of actors in Berland and the number of Vasya's favourite actors. 
	The second line contains k distinct integers ai (1 ≤ ai ≤ m) — the identifiers of Vasya's favourite actors.
	The third line contains a single integer n (1 ≤ n ≤ 100) — the number of movies in Vasya's list.
	Then follow n blocks of lines, each block contains a movie's description. The i-th movie's description contains three lines: 
	  the first line contains string si (si consists of lowercase English letters and can have the length of from 1 to 10 characters, inclusive) — the movie's title,  the second line contains a non-negative integer di (1 ≤ di ≤ m) — the number of actors who starred in this movie, the third line has di integers bi, j (0 ≤ bi, j ≤ m) — the identifiers of the actors who star in this movie. If bi, j = 0, than Vasya doesn't remember the identifier of the j-th actor. It is guaranteed that the list of actors for a movie doesn't contain the same actors. All movies have distinct names. The numbers on the lines are separated by single spaces.

	Output
	Print n lines in the output. In the i-th line print: 
	  0, if the i-th movie will surely be the favourite;  1, if the i-th movie won't surely be the favourite;  2, if the i-th movie can either be favourite, or not favourite. 
	Examples
	Input
	5 3
	1 2 3
	6
	firstfilm
	3
	0 0 0
	secondfilm
	4
	0 0 4 5
	thirdfilm
	1
	2
	fourthfilm
	1
	5
	fifthfilm
	1
	4
	sixthfilm
	2
	1 0

	Output
	2
	2
	1
	1
	1
	2

	Input
	5 3
	1 3 5
	4
	jumanji
	3
	0 0 0
	theeagle
	5
	1 2 3 4 0
	matrix
	3
	2 4 0
	sourcecode
	2
	2 4

	Output
	2
	0
	1
	1


	Note
	Note to the second sample: 
	  Movie jumanji can theoretically have from 1 to 3 Vasya's favourite actors.  Movie theeagle has all three favourite actors, as the actor Vasya failed to remember, can only have identifier 5.  Movie matrix can have exactly one favourite actor.  Movie sourcecode doesn't have any favourite actors. Thus, movie theeagle will surely be favourite, movies matrix and sourcecode won't surely be favourite, and movie jumanji can be either favourite (if it has all three favourite actors), or not favourite.


	'''

	print(q)
	q = q.replace('\r\n', '\n').replace('\n', ' ¶ ').replace('¶  ¶', '¶ ¶').rstrip(' ¶').rstrip(' ').rstrip('¶').replace('† ', '† ').replace(' ‡', ' ‡')
	print(q)
	toked = question_to_tokenized_fields(q)
	print(toked)
	for i in toked:
		print(i)
		for j in i:
			print j

