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

def escape_lt(html):
    html_list = list(html)
    for index in xrange(0, len(html) - 1):
        if html_list[index] == '<' and html_list[index + 1] == ' ':
            html_list[index] = '&lt;'
    return ''.join(html_list)


def get_solution_ids(name, language):

	if language == 'python':
		#FIBQ
		url = 'https://www.codechef.com/status/%s?sort_by=All&sorting_order=asc&language=4&status=15&handle=&Submit=GO' % (name)
		url2 = 'https://www.codechef.com/status/%s?page=1&sort_by=All&sorting_order=asc&language=4&status=15&handle=&Submit=GO' % (name)
	elif language == 'c++':
		url = 'https://www.codechef.com/status/%s?sort_by=All&sorting_order=asc&language=41&status=15&handle=&Submit=GO' % (name)
		url2 = 'https://www.codechef.com/status/%s?page=1&sort_by=All&sorting_order=asc&language=41&status=15&handle=&Submit=GO' % (name)
	else:
		pass

	page1 = requests.get(url)
	if str(page1) == "<Response [503]>":
		while str(page1) == "<Response [503]>":
			time.sleep(1)
			page1 = requests.get(url)

	page2 = requests.get(url2)
	if str(page2) == "<Response [503]>":
		while str(page2) == "<Response [503]>":
			time.sleep(1)
			page2 = requests.get(url2)

	html_content = page1.text + page2.text

	messages = []

	solution_id = re.findall("href='/viewsolution/(.+?)' target='_blank'>View", html_content)
	pts = re.findall("/>\\[(.+?)pts\\]<", html_content)

	solution_ids = []

	if len(pts) != 0 and len(solution_id) != 0:

		for i in range(len(pts)):
			messages.append([str(solution_id[i]), str(pts[i])])

	return messages

def download_descriptions_solutions(filename, index_n):
	#root_dir = 'codechef_alter_data'
	root_dir = 'codechef_pts_data'

	file = open(filename, 'r')
	f = open(filename, 'r')

	index_n_int = int(index_n)

	start = index_n_int + (500*index_n_int)
	end = start + 499

	easy = []
	medium = []
	hard = []
	harder = []
	hardest = []
	external = []

	g = ""
	i=0
	for line in f:
		if str(line).find('type=') != -1:
			body = re.search('type=(.*)', line)
			g = body.group(1)
		else:
			if str(g) == "easy":
				easy = eval(line)
			elif str(g) == "medium":
				medium = eval(line)
			elif str(g) == "hard":
				hard = eval(line)
			elif str(g) == "harder":
				harder = eval(line)
			elif str(g) == "hardest":
				hardest = eval(line)
			elif str(g) == "external":
				external = eval(line)
			else:
				pass

	all_names = []
	all_names_p = []
	all_names =[["easy", easy], ["medium", medium], ["hard", hard], ["harder", harder], ["hardest", hardest], ["external", external]]

	for ndx, n in enumerate(all_names):
		category = all_names[ndx][0]
		problem_list = all_names[ndx][1]

		language = ["python", "c++"]

		for idx, i in enumerate(problem_list):

			if not os.path.exists(root_dir):
			    os.makedirs(root_dir)

			cat_dir = root_dir + "/" + category

			if not os.path.exists(cat_dir):
			    os.makedirs(cat_dir)

			save_dir = cat_dir

			if not os.path.exists(save_dir):
				os.makedirs(save_dir)

			ids_l = []
			for l in language:
				ids = get_solution_ids(i, l)
				problem2_id_pts = {}
				problem2_id_pts[i + '/' + 'solutions_' + l] = ids

				print(problem2_id_pts)
				print(len(ids))

				s_file_path = 'tags' + '_' + category
				if len(ids) != 0:
					solution_file = open(s_file_path, 'a')
					solution_file.write(str(problem2_id_pts) + '\n')

			print(problem2_id_pts)

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=str, default="index", help='')
args = parser.parse_args()

index_n = args.index

#get_solution_ids('codechef_problem_names.txt', index_n)

download_descriptions_solutions('codechef_problem_names.txt', index_n)






