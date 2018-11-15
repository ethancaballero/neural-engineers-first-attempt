# -*- coding: utf-8 -*-
import shutil
import os
import re
import requests
import urllib2
from pprint import pprint
import bs4
from bs4 import BeautifulSoup
import html2text
import time
import argparse
import datetime

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'

def get_request(url, headers={}):
    response = requests.get(url,
                                headers=headers)

    return response

def get_submissions():
    handle = 'prashantpandeyfun10'

    url = "https://www.hackerearth.com/submissions/" + handle
    t = get_request(url)
    if t == -1 or t == {}:
        return t

    tmp_string = t.headers["set-cookie"]
    csrf_token = re.findall(r"csrftoken=\w*", tmp_string)[0][10:]

    response = {}
    response["host"] = "www.hackerearth.com"
    response["user-agent"] = user_agent
    response["accept"] = "application/json, text/javascript, */*; q=0.01"
    response["accept-language"] = "en-US,en;q=0.5"
    response["accept-encoding"] = "gzip, deflate"
    response["content-type"] = "application/x-www-form-urlencoded"
    response["X-CSRFToken"] = csrf_token
    response["X-Requested-With"] = "XMLHttpRequest"
    response["Referer"] = "https://www.hackerearth.com/submissions/" + handle + "/"
    response["Connection"] = "keep-alive"
    response["Pragma"] = "no-cache"
    response["Cache-Control"] = "no-cache"
    response["Cookie"] = tmp_string

    it = 1
    submissions = {handle: {}}
    for page_number in xrange(1, 5):
        print(page_number)
        submissions[handle][page_number] = {}
        url = "https://www.hackerearth.com/AJAX/feed/newsfeed/submission/user/" + handle + "/?page=" + str(page_number)

        tmp = get_request(url, headers=response)

        if tmp.status_code != 200:
            return -1

        json_response = tmp.json()

        if json_response["status"] == "ERROR":
            break

        body = json_response["data"]
        soup = bs4.BeautifulSoup(body, "lxml")

        trs = soup.find("tbody").find_all("tr")
        for tr in trs:

            submissions[handle][page_number][it] = []
            submission = submissions[handle][page_number][it]
            append = submission.append

            all_tds = tr.find_all("td")
            all_as = tr.find_all("a")
            time_stamp = all_tds[-1].contents[1]["title"]
            time_stamp = time.strptime(str(time_stamp), "%Y-%m-%d %H:%M:%S")

            # Time of submission
            time_stamp = datetime.datetime(time_stamp.tm_year,
                                           time_stamp.tm_mon,
                                           time_stamp.tm_mday,
                                           time_stamp.tm_hour,
                                           time_stamp.tm_min,
                                           time_stamp.tm_sec) + \
                                           datetime.timedelta(minutes=630)
            curr = time.strptime(str(time_stamp), "%Y-%m-%d %H:%M:%S")

            append(str(time_stamp))

            # Problem Name/URL
            problem_link = "https://www.hackerearth.com" + all_as[1]["href"]
            append(problem_link)
            problem_name = all_as[1].contents[0]
            append(problem_name)

            # Status
            try:
                status = all_tds[2].contents[1]["title"]
            except IndexError:
                status = "Others"

            if status.__contains__("Accepted"):
                status = "AC"
            elif status.__contains__("Wrong"):
                status = "WA"
            elif status.__contains__("Compilation"):
                status = "CE"
            elif status.__contains__("Runtime"):
                status = "RE"
            elif status.__contains__("Memory"):
                status = "MLE"
            elif status.__contains__("Time"):
                status = "TLE"
            else:
                status = "OTH"
            append(status)

            # Points
            if status == "AC":
                points = "100"
            else:
                points = "0"
            append(points)

            # Language
            language = all_tds[5].contents[0]
            append(language)

            # View Link
            if len(all_as) == 4:
                append("https://www.hackerearth.com" + all_as[-2]["href"])
            else:
                append("")

            it += 1

    return submissions

print get_submissions()