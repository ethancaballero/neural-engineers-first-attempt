# -*- coding: utf-8 -*-
import shutil
import os
import re
import requests
import urllib2
from pprint import pprint
from bs4 import BeautifulSoup
import html2text
import time
import argparse
import concurrent.futures

cookies = dict()
cookies['tcsso'] = '40451530|b0be8a6e3acae9d8743c91ada7294a5b65a698b0dfa82cda539d54a7d41e7584'


def sub_strip(matchobj):   
    return matchobj.group(0).replace(u"\u2009", "")

def get_problem_list(url):
	page = requests.get(url)
	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url)
	html_content = page.text

	soup = BeautifulSoup(html_content, "html.parser") # making soap

	messages = []

	text = soup.select("body a")

	for row in text:
		message = ""
		raw = str(row)
		body = re.search(' href="/problemset/problem/(.*)">', raw)

		if body != None:
			w = body.group(1)
			message = str(w)
			c = message.split('/')
			messages.append(c)

	return messages

def get_rounds(url):
	page = requests.get(url, cookies=cookies)
	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url, cookies=cookies)
	html_content = page.text

	round_content = re.search('Select a Round:</OPTION>(.+?)Select a Room:</OPTION>', html_content.replace("\\", ""), flags=re.S)

	body = re.findall('<OPTION value="/stat\\?c=room_stats(.+?)</OPTION>', round_content.group(1).replace("\\", ""), flags=re.S)

	rounds = []

	for sub_body in body:
		message = str(sub_body)
		if re.search('">', message, flags=re.S):
			c = message.split('">')
		else:
			c = message.split('" selected>')
		rounds.append(c)

	return rounds

def get_rooms(url):
	page = requests.get(url, cookies=cookies)
	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url, cookies=cookies)
	html_content = page.text

	room_content = re.search('Select a Room:</OPTION>(.+?)COLSPAN="4" CLASS="statText"', html_content.replace("\\", ""), flags=re.S)
	
	body = re.findall('<OPTION value="/stat\\?c=room_stats(.+?)</OPTION>', room_content.group(1).replace("\\", ""), flags=re.S)

	rooms = []

	for sub_body in body:
		message = str(sub_body)
		if re.search('">', message, flags=re.S):
			c = message.split('">')
		else:
			c = message.split('" selected>')
		rooms.append(c)

	return rooms

def get_authors(url1):
	page = requests.get(url1, cookies=cookies)
	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url1, cookies=cookies)
	author_content = page.text

	body_score = re.findall('module=MemberProfile(.+?)\\.[0-9][0-9]</TD>', author_content.replace("\\", ""), flags=re.S)

	authors = []

	for i in body_score:
		if re.search('ALIGN="right">(.+?)</TD>', i) != str(0):
			author = re.search('(.+?)" CLASS', i)
			authors.append(author.group(1))

	return authors


def get_solution_ids(url2):
	page = requests.get(url2, cookies = cookies)
	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url2)
	html_content = page.text

	body_score = re.findall('<TD CLASS="statText" HEIGHT="13">(.+?)<TD COLSPAN="8"><IMG SRC="/i/clear.gif" ALT="" WIDTH="1" HEIGHT="3" BORDER="0"></TD>', html_content.replace("\\", ""), flags=re.S)

	solution_ids_sub = []

	for i in body_score:
		if "Passed System Test" in i:
			solution_id = re.search('&pm=(.+?)&cr', i)
			'''MAYBE ADD SOME CODE HERE TO GET NAME AND NOT JUST NUMBER REPRESENTING NAME'''
			solution_ids_sub.append(solution_id.group(1))

	return solution_ids_sub

def get_samples(url2):

	page = requests.get(url2, cookies=cookies)
	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url2, cookies=cookies)
	html_content = page.text

	body = re.findall('<TR VALIGN="top" class="alignTop">(.+?)CLASS="statText" ALIGN="right">Passed</TD>', html_content, flags=re.S)

	inputs = []
	outputs = []

	for i in body:
		inp = re.search('CLASS="statText" ALIGN="left">(.+?)</TD>', i)
		out = re.search('CLASS="statText" ALIGN="right">(.+?)</TD>', i)
		inputs.append(inp.group(1))
		outputs.append(out.group(1))

	return solution



def get_pages():
	all_rounds = get_rounds('https://community.topcoder.com/stat?c=room_stats&rd=16775&rm=329101')
	for i in all_rounds:
		url = 'https://community.topcoder.com/stat?c=room_stats' + i[0]
		for j in get_rooms(url):
			url1 = 'https://community.topcoder.com/stat?c=room_stats' + j[0]
			if "Division-II" in str(j[1]):
				for k in get_authors(url1):
					url2 = "https://community.topcoder.com/stat?c=coder_room_stats" + j[0] + k
					solution_ids = get_solution_ids(url2)


					'''get_sample PROBABLY BELONGS ELSEWHERE'''
					url_sample = "https://community.topcoder.com/stat?c=problem_solution" + j[0] + k + "&pm=" + solution_ids[0]
					get_samples(url_sample)
					

					for m in get_solution_ids(url2):
						url3 = "https://community.topcoder.com/stat?c=problem_solution" + j[0] + k + "&pm=" + m
						get_solution(url3)

			else:
				for k in get_authors(url1):
					url2 = "https://community.topcoder.com/stat?c=coder_room_stats" + j[0] + k

def get_description(i):
	descriptions = []
	left_out = []
	failed_to_download_d = []

	url = i

	page = requests.get(url)

	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url)

	html_content = page.text

	if re.search('"message":"requests limit exhausted"', html_content) != None:
		while re.search('message":"requests limit exhausted', html_content) != None:
			time.sleep(1)
			page = requests.get(url)
			html_content = page.text



	if html_content==None:
		failed_to_download_d.append(i)

	if re.search('img src="http://www.topcoder.com/contest/problem/', html_content.replace("\\", "")) == None and re.search('src="http://espresso.codeforces.com', html_content.replace("\\", "")) == None and re.search('"message":"Problem is not visible now. Please try again later."', html_content) == None and re.search('Statement is not available', html_content) == None:

		body = re.findall('Problem Statement</h3></td></tr><tr>(.+?)</td></tr></table><hr><p>', html_content, flags=re.S)

		w = body[0]
		w = w.replace('class="upper-index">', 'class="upper-index">^')
		w = w.replace("&#160;&#160;&#160;&#160;", "")
		w = w.replace(':</td><td class="statText">', ':    </td><td class="statText">')

		'''NEED TO PUT PUT CODE HERE TO REMOVE SPACES IN NEGATIVE EXPONENTS'''
		w = re.sub('class="upper-index">(.+?)</sup>', sub_strip, w, re.S)

		w = w.replace("</td></tr><tr>", "\n</td></tr><tr>")
		w = w.replace("<ul>\n", "<ul>")
		w = w.replace("<li>", u"• ")
		w = w.replace("</li>\n</ul>", "</li></ul>")
		w = w.replace('<td align="center" valign="top" class="statText">-', '<td align="center" valign="top" class="statText">- ')

		w = BeautifulSoup(w, "html.parser").get_text()
		w = w.replace("All submissions for this problem are available.", "")

		w = re.sub('Read problems statements in (.+?)\\\\n', '', w, re.M)
		w = re.sub('Subtasks(.+?)Example', 'Example', w, re.S)

		w = w.replace("\u003C","<")
		w = w.replace("\u003E",">")

		w = w.replace(u"\u00A0","\n")

		w = re.sub("\n\n\n\n\n\n", "\n\n\n\n\n", w, flags=re.M)
		w = re.sub("\n\n\n\n\n", "\n\n\n\n", w, flags=re.M)
		w = re.sub("\n\n\n\n", "\n\n\n", w, flags=re.M)

		w = re.sub('([^\n])(\n[0-9])(\\))(\n)', '\n\\2)\n', w, flags=re.M)
		w = re.sub('(Example\n)(\n[0-9])(\\))(\n)', 'Example\\2)\n', w, flags=re.M)

		w = w.replace("\\","\\\\")

		descriptions.append(w.encode('utf-8').decode('string-escape'))
	else:
		left_out.append(i)


	return descriptions, left_out, failed_to_download_d


def get_solutions(contest, solution_ids):
	solutions = {}
	with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
		future_to_url = {executor.submit(get_solution, contest, i): i for i in solution_ids}
		for future in concurrent.futures.as_completed(future_to_url):
			data = future.result()

			if data[2] == None:
				solutions[data[0]] = data[1]

	return solutions

def get_solution(url):
	page = requests.get(url, cookies=cookies)
	if str(page) == "<Response [503]>":
		while str(page) == "<Response [503]>":
			time.sleep(1)
			page = requests.get(url, cookies=cookies)
	html_content = page.text

	body = re.findall('<TD CLASS="problemText" COLSPAN="8" VALIGN="middle" class="alignMiddle" ALIGN="left">\n            (.+?)<BR>\n        </TD>', html_content, flags=re.S)

	text = body[0]

	text = text.replace("<BR>","\n")

	failed_to_download = None
	solution = None


	if len(text)==0:
		failed_to_download = solution_id
	else:
		body = BeautifulSoup(str(text), "html.parser").get_text()

		body = body.replace("\\","\\\\")
		solution = body.encode('utf-8').decode('string-escape')

	return solution

def download_all_challenge_names(filename):
	target = open(filename, 'w')

	problem_list = []

	for i in range(0,30):
		a = 'http://codeforces.com/problemset/page/' + str(i+1)
		l = get_problem_list(a)
		for jdx, j in enumerate(l):
			if jdx % 2 == 0:
				problem_list.append(j)
	target.write(str(problems))

def download_descriptions_solutions(filename, index_n):
	root_dir = 'codeforces_data'

	file = open(filename, 'r')
	f = open(filename, 'r')

	index_n_int = int(index_n)

	start = index_n_int + (600*index_n_int)
	end = start + 599

	all_names = []

	for line in f:
		raw = eval(str(line))

	a = ""
	b = ""

	all_names = raw
	language = ["python", "c++"]

	for idx, i in enumerate(all_names):

		descriptions, left_out, failed_to_download_d = get_description(i)
		if i not in left_out:
			if not os.path.exists(root_dir):
			    os.makedirs(root_dir)

			save_dir = root_dir + "/" + i[0] + "_" + i[1]

			if not os.path.exists(save_dir):
			    os.makedirs(save_dir)

			description_dir = save_dir + "/description"

			if not os.path.exists(description_dir):
			    os.makedirs(description_dir)

			description_file_path = description_dir + "/description.txt"
			description_file = open(description_file_path, 'w')
			description_file.write(descriptions[0])

			ids_l = []
			for l in language:
				ids = get_solution_ids(i, l)
				ids_l.append(ids)

				solutions = get_solutions(i, ids)

				solution_dir = save_dir + "/solutions_" + l

				if not os.path.exists(solution_dir):
				    os.makedirs(solution_dir)

				for jdx, j in enumerate(solutions):
					if len(solutions[j]) < 10000:
						solution_file_path = solution_dir + "/" + j + ".txt"
						solution_file = open(solution_file_path, 'w')
						solution_file.write(solutions[j])


			if len(ids_l[0]) == 0 and len(ids_l[1]) == 0:
				shutil.rmtree(save_dir)

	print("Finished download process")
	if len(failed_to_download) > 0:
	    print("Following challenges failed to download: " + str(failed_to_download))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--index', type=str, default="1", help='')
	args = parser.parse_args()

	index_n = args.index

	#download_all_challenge_names('challenges_all.txt')
	#download_descriptions_solutions('challenges_all.txt', index_n)

	#url = "https://community.topcoder.com/stat?c=problem_statement&pm=14346&rd=16790"
	#url = "https://community.topcoder.com/stat?c=problem_statement&pm=14368"
	#url = 'https://community.topcoder.com/stat?c=problem_solution&rm=329103&rd=16775&pm=14340&cr=23089515'
	url = 'https://community.topcoder.com/stat?c=room_stats&rd=16775&rm=329100'

	#get_description(url)
	#get_solution(url)
	#get_rounds(url)
	#get_rooms(url)
	get_pages()









