# -*- coding: utf-8 -*-

import string

import segtok.segmenter
import segtok.tokenizer

from segtok.tokenizer import symbol_tokenizer, word_tokenizer, web_tokenizer
from segtok.tokenizer import split_possessive_markers, split_contractions

'''ADD SECTION FOR CONVERTING EVERYTHING TO LOWER CASE'''

#q = '''Once when Gerald studied in the first year at school , his teacher gave the class the following homework . She offered the students a string consisting of n small Latin letters ; the task was to learn the way the letters that the string contains are written . However , as Gerald is too lazy , he has no desire whatsoever to learn those letters . That &apos;s why he decided to lose some part of the string ( not necessarily a connected part ) . The lost part can consist of any number of segments of any length , at any distance from each other . However , Gerald knows that if he loses more than k characters , it will be very suspicious . ¶ Find the least number of distinct characters that can remain in the string after no more than k characters are deleted . You also have to find any possible way to delete the characters . ¶ ¶ Input ¶ The first input data line contains a string whose length is equal to n ( 1 ≤ n ≤ 10 ^ 5 ) . The string consists of lowercase Latin letters . The second line contains the number k ( 0 ≤ k ≤ 10 ^ 5 ) . ¶ ¶ Output ¶ Print on the first line the only number m — the least possible number of different characters that could remain in the given string after it loses no more than k characters . ¶ Print on the second line the string that Gerald can get after some characters are lost . The string should have exactly m distinct characters . The final string should be the subsequence of the initial string . If Gerald can get several different strings with exactly m distinct characters , print any of them . ¶ ¶ Examples ¶ Input ¶ aaaaa ¶ 4 ¶ ¶ Output ¶ 1 ¶ aaaaa ¶ ¶ Input ¶ abacaba ¶ 4 ¶ ¶ Output ¶ 1 ¶ aaaa ¶ ¶ Input ¶ abcdefgh ¶ 10 ¶ ¶ Output ¶ 0 ¶ ¶ ¶ ¶ Note ¶ In the first sample the string consists of five identical letters but you are only allowed to delete 4 of them so that there was at least one letter left . Thus , the right answer is 1 and any string consisting of characters &quot; a &quot; from 1 to 5 in length . ¶ In the second sample you are allowed to delete 4 characters . You cannot delete all the characters , because the string has length equal to 7 . However , you can delete all characters apart from &quot; a &quot; ( as they are no more than four ) , which will result in the &quot; aaaa &quot; string . ¶ In the third sample you are given a line whose length is equal to 8 , and k = 10 , so that the whole line can be deleted . The correct answer is 0 and an empty string . ¿'''
#q = '''Once when Gerald studied in the first year at school, his teacher gave the class the following homework. She offered the students a string consisting of n small Latin letters; the task was to learn the way the letters that the string contains are written. However, as Gerald is too lazy, he has no desire whatsoever to learn those letters. That &apos;s why he decided to lose some part of the string ( not necessarily a connected part ). The lost part can consist of any number of segments of any length, at any distance from each other. However, Gerald knows that if he loses more than k characters, it will be very suspicious. ¶ Find the least number of distinct characters that can remain in the string after no more than k characters are deleted. You also have to find any possible way to delete the characters. ¶ ¶ Input ¶ The first input data line contains a string whose length is equal to n ( 1 ≤ n ≤ 10 ^ 5 ). The string consists of lowercase Latin letters. The second line contains the number k ( 0 ≤ k ≤ 10 ^ 5 ). ¶ ¶ Output ¶ Print on the first line the only number m — the least possible number of different characters that could remain in the given string after it loses no more than k characters. ¶ Print on the second line the string that Gerald can get after some characters are lost. The string should have exactly m distinct characters. The final string should be the subsequence of the initial string. If Gerald can get several different strings with exactly m distinct characters, print any of them. ¶ ¶ Examples ¶ Input ¶ aaaaa ¶ 4 ¶ ¶ Output ¶ 1 ¶ aaaaa ¶ ¶ Input ¶ abacaba ¶ 4 ¶ ¶ Output ¶ 1 ¶ aaaa ¶ ¶ Input ¶ abcdefgh ¶ 10 ¶ ¶ Output ¶ 0 ¶ ¶ ¶ ¶ Note ¶ In the first sample the string consists of five identical letters but you are only allowed to delete 4 of them so that there was at least one letter left. Thus, the right answer is 1 and any string consisting of characters &quot; a &quot; from 1 to 5 in length. ¶ In the second sample you are allowed to delete 4 characters. You cannot delete all the characters, because the string has length equal to 7. However, you can delete all characters apart from &quot; a &quot; ( as they are no more than four ), which will result in the &quot; aaaa &quot; string. ¶ In the third sample you are given a line whose length is equal to 8, and k = 10, so that the whole line can be deleted. The correct answer is 0 and an empty string.'''

q = '''Description: ¶ Name string is a string consisting of letters "R","K" and "V". Today Oz wants to design a name string in a beautiful manner. Actually Oz's cannot insert these three letters arbitrary anywhere ,he has to follow some rules to make the name string look beautiful. First thing is that the name string should consist of at most two different letters. Secondly adjacent letters in name string must be different. ¶ ¶ After this procedure Oz wants name string to be as long as possible. Given the number of "R","K" and "V" letters that you have initially ,help Oz to find the maximum length of name string that Oz can make. ¶ ¶ Input : ¶ The first line contains the number of test cases T. Each test case consists of three space separated integers - A,B and C representing number of "R" letters, number of "K" letters and number of "V" letters respectively.  ¶ ¶ Output : ¶ For each test case T, output maximum length of name string that Oz can make. ¶ ¶ Constraints : ¶ 1 ≤ T ≤100 ¶ 0 ≤ A,B,C ≤10^6 ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 1 2 asfas 5 ¶ 0 0 2 ¶ ¶ SAMPLE OUTPUT ¶ 5 ¶ 1 ¶ ¶ Explanation ¶ ¶ For first sample : ¶ The longest name string possible is :  VKVKV  using 3 "V" letters and 2 "K" letters and its length is 5.'''
#q = '''Description: ¶ Name string is a string consisting of letters "R","K" and "V" . Today Oz wants to design a name string in a beautiful manner . Actually Oz cannot insert these three letters arbitrary anywhere ,he has to follow some rules to make the name string look beautiful . First thing is that the name string should consist of at most two different letters . Secondly adjacent letters in name string must be different . ¶ ¶ After this procedure Oz wants name string to be as long as possible . Given the number of "R","K" and "V" letters that you have initially , help Oz to find the maximum length of name string that Oz can make . ¶ ¶ Input : ¶ The first line contains the number of test cases T . Each test case consists of three space separated integers - A,B and C representing number of "R" letters, number of "K" letters and number of "V" letters respectively .  ¶ ¶ Output : ¶ For each test case T, output maximum length of name string that Oz can make . ¶ ¶ Constraints : ¶ 1 ≤ T ≤100 ¶ 0 ≤ A,B,C ≤10^6 ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 1 2 5 ¶ 0 0 2 ¶ ¶ SAMPLE OUTPUT ¶ 5 ¶ 1 ¶ ¶ Explanation ¶ ¶ For first sample : ¶ The longest name string possible is :  VKVKV  using 3 "V" letters and 2 "K" letters and its length is 5 .'''

#"""
#q = '''Codex is about to start and Ram has not done dinner yet. So, he quickly goes to hostel mess and finds a long queue in front of  food counter. But somehow he manages to take the food plate and reaches in front of the queue. The plates are divided into sections such that it has 2 rows and N columns. ¶ ¶ Due to the crowd Ram knows that when he will get out of the queue the food will get mixed. So, he don't want to put food in two consecutive sections column wise ¶ but he can put food in two consecutive sections row wise as spacing between the rows is good enough to take food out of the queue safely. If he doesn't like the food, he will not take food. You are given N and you have to tell the number of ways in which food can be taken without getting it mixed. ¶ ¶ Input Format: ¶ First line contains T which denotes number of test cases and each test case represents a single line containing the value of N.  ¶ ¶ Output Format ¶ Output the total ways for each input. ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 1 ¶ 3 ¶ ¶ SAMPLE OUTPUT ¶ 4 ¶ 25 ¶ ¶ Explanation ¶ ¶ Explanation: ¶ ¶ Case 1: ¶ Plate has 2 rows and 1 column each. So, Ram can ¶ Put food in upper section. ¶ Put food in lower section. ¶ Put food in both section. ¶ Do Not food in either section. ¶ ¶ Case 2: ¶ Plate has 2 rows and 3 columns. So, possible ways for one row are PNN, PNP, NNN, NPN, NNP where P represents food taken and N represents food not taken. ¶ Total possible ways are 25 because a way to put food in 1 row can correspond  ¶ to any of 5 ways on other row. '''
#q = '''Rani works at sweet shop and is currently arranging sweets i.e. putting sweets in boxes . There are some boxes which contains sweets initially but they are not full . Rani have to fill boxes completely thereby minimizing the total no. of boxes . ¶ You are given N no. of boxes, their initial amount and total capacity in grams . You have to print out the minimum no. of boxes required to store given amount of sweets . ¶ ¶ Input: ¶ First line contains no. of test cases . Each test case contains N no. of boxes and next two lines contains  n values each denoting their initial amount and total capacity . ¶ Output: ¶ Print out the minimum no. of boxes required to store given amount of sweets . ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 3 ¶ 300 525 110 ¶ 350 600 115 ¶ 6 ¶ 1 200 200 199 200 200 ¶ 1000 200 200 200 200 200 ¶ ¶ SAMPLE OUTPUT ¶ 2 ¶ 1 ¶ ¶ Explanation ¶ ¶ Testcase 1: ¶ One way to pack the sweets onto as few boxes as possible is as follows . First, move 50 g from box 3 to box 1, completely filling it up . Next, move the remaining 60 g from box 3 to box 2 . There are now two boxes which contain sweets after this process, so your output should be 2 . ¶ Testcase 2: ¶ One way to consolidate the sweets would be to move the 1 g from box 1 to box 4 . However, this is a poor choice, as it results in only one free box and five boxes which still contain sweets . A better decision is to move all the sweets from the other five boxes onto box 1 . Now there is only one box which contains sweets . Since this is the optimal strategy, the output should be 1 . '''

#q = '''Rani works at sweet shop and is currently arranging sweets i.e. putting sweets in boxes. There are some boxes which contains sweets initially...but they are not full. Rani have to fill boxes completely thereby minimizing the total no. of boxes. ¶ You are given N no. of boxes, their initial amount and total capacity in grams. You have to print out the minimum no. of boxes required to store given amount of sweets. ¶ ¶ Input: ¶ First line contains no. of test cases. Each test case contains N no. of boxes and next two lines contains  n values each denoting their initial amount and total capacity. ¶ Output: ¶ Print out the minimum no. of boxes required to store given amount of sweets. ¶ ¶ SAMPLE INPUT ¶ 2 ¶ 3 ¶ 300 525 110 ¶ 350 600 115 ¶ 6 ¶ 1 200 200 199 200 200 ¶ 1000 200 200 200 200 200 ¶ ¶ SAMPLE OUTPUT ¶ 2 ¶ 1 ¶ ¶ Explanation ¶ ¶ Testcase 1: ¶ One way to pack the sweets onto as few boxes as possible is as follows. First, move 50 g from box 3 to box 1, completely filling it up. Next, move the remaining 60 g from box 3 to box 2. There are now two boxes which contain sweets after this process, so your output should be 2. ¶ Testcase 2: ¶ One way to consolidate the sweets would be to move the 1 g from box 1 to box 4. However, this is a poor choice, as it results in only one free box and five boxes which still contain sweets. A better decision is to move all the sweets from the other five boxes onto box 1. Now there is only one box which contains sweets. Since this is the optimal strategy, the output should be 1.'''
#"""

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

#"""
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
#"""

'''
print q
q = q.replace('\r\n', '\n').replace('\n', ' ¶ ').replace('¶  ¶', '¶ ¶').rstrip(' ¶').rstrip(' ').rstrip('¶').replace('† ', '† ').replace(' ‡', ' ‡')
print q
toked = question_to_tokenized_fields(q)
print toked
for i in toked:
	print i
	#for j in i:
		#print j
	#'''

