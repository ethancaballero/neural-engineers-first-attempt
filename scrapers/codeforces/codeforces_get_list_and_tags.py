
#from bs4 import BeautifulSoup

from pprint import pprint
from bs4 import BeautifulSoup
import requests
import urllib2
import re

'''
info = 'http://codeforces.com/api/user.status?handle=tacklemore&from=1&count=1'
solution = 'view-source:http://codeforces.com/contest/686/submission/18671530'
do not include problems with http://codeforces.com/predownloaded/
'''

def get_problem_list(url):
	page = requests.get(url)
	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url)
	html_content = page.text

	soup = BeautifulSoup(html_content, "html.parser") # making soap

	messages = []
	tags = []
	problem_and_tags = {}
	problem_and_tags_array = []

	text = soup.select("body a")
	body_problem_prev = None
	b_p = None

	for row in text:
		message = ""
		raw = str(row)
		body_problem = re.search(' href="/problemset/submit/(.*)">', raw)
		body_tag = re.search(' href="/problemset/tags/(.*)" style', raw)
		#second_tag = re.search('style="float:right', raw)

		if body_problem != None:
			w = body_problem.group(1)
			message = str(w)
			b_p = message.replace('/', '_')
			problem_and_tags[b_p] = tags
			problem_and_tags_array.append(problem_and_tags)
			problem_and_tags = {}
			tags = []

		if body_tag != None:
			w = body_tag.group(1)
			message = str(w)
			b_t = message
			tags.append(b_t)

	return problem_and_tags_array


problem_list = []

for i in range(0,30):
	a = 'http://codeforces.com/problemset/page/' + str(i+1)
	l = get_problem_list(a)
	problem_list += l


print(problem_list)
'''
for k in sorted(problem_list):
   print k.replace(' ', '_'), problem_list[k]
   #'''
description_file = open("tags.txt", 'w')
description_file.write('')

for k in problem_list:
	description_file = open("tags.txt", 'a')
	description_file.write(str(k) + "\n")