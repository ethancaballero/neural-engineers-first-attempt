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
from sys import argv
import time
from selenium import webdriver

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'


def get_request(url, headers={}):
    response = requests.get(url,
                                headers=headers)

    return response

def post_request(url, data={}, headers={}, cookies={}):
    response = requests.post(url,
                            data=data,
                            headers=headers,
                            cookies=cookies)

    return response

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
        body = re.search('/problem/algorithm/(.*)/">', raw)

        if body != None:
            w = body.group(1)
            message = str(w)
            c = message
            #if message != 'easy' and message != 'medium' and message != 'hard' and message != 'challenge' and message != 'extcontest' and message != 'school':
            messages.append(c)

    return messages

def get_submissions():
    page = get_request(url, headers=response)

    html_content = page.text

    text = soup.select("body a")

    body = re.search('<a href="/problem/algorithm/(.*)/" class="link-13 track-problem-link">', html_content)

    w = body.group(1)

    lordoftherings_value = '13c082ac336859d586aa5364c086d26f:44751f02ffbb8d82fb3deddca4da60de'

    cookies = dict()
    cookies["lordoftherings"] = lordoftherings_value

    browser=webdriver.Chrome()

    cookie = {'name': "lordoftherings", 'value' : 'a5dr3g48ag2dg8s2b8r57gkil6ioip74:7c34ac7cc9b2c971eafaba58840e0717', 'path' : '/'}

    url_home_page = 'https://www.hackerearth.com/submission/4440655/'

    url2 = 'https://www.wikipedia.org/'

    browser.get(url_home_page)                                                    # This opens a firefox console  
    browser.implicitly_wait(20)

    login_but=browser.find_element_by_xpath("//li[contains(@class,'nav-bar-menu login-menu-btn')]")

    webdriver.ActionChains(browser).click(login_but).perform()

    url = 'https://www.hackerearth.com/submission'

    username = browser.find_element_by_id("id_login")
    password = browser.find_element_by_id("id_password")

    username.send_keys("evc123@att.net")
    password.send_keys("8659vlec")

    browser.find_element_by_name("submit").click()

    unlock_but=browser.find_element_by_xpath("//a[contains(@class,'button btn-blue ajax-get') and .//text()='Unlock it']")

    webdriver.ActionChains(browser).click(unlock_but).perform()

    sdfg

    handle = 'prashantpandeyfun10'

    name = 'algorithm/karan-and-even-numbers-1'    

    url = "https://www.hackerearth.com/submission/4440655/"

    url = "https://www.hackerearth.com/problem/" + name + "/activity/"
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
    response["Referer"] = url
    response["Connection"] = "keep-alive"
    response["Pragma"] = "no-cache"
    response["Cache-Control"] = "no-cache"
    response["Cookie"] = tmp_string

    it = 1
    submissions = {handle: {}}
    for index_number in xrange(1, 5):
        submissions[handle][index_number] = {}

        url_post = "https://www.hackerearth.com/AJAX/algorithm/42373/unlock-problem-submission/"

        url_auth = 'https://www.hackerearth.com/realtime/pusher/auth/'

        data = {'csrf_token':csrf_token, 'action':'setupSubmissionFilter', 'frameProblemIndex':'A', 'verdictName':'OK'}

        url_auth = 'https://www.hackerearth.com/realtime/pusher/auth/'

        idk = post_request(url_post, headers=response)

        url = "https://www.hackerearth.com/submission/4440655/"

        page = get_request(url, headers=response)

        html_content = page.text

        soup = BeautifulSoup(html_content, "html.parser")

        body = re.search('/submission/key/(.*)/', html_content)

        w = body.group(1)

        return w


url = 'https://www.hackerearth.com/problems/'
url_college = 'https://www.hackerearth.com/college-challenges/problems/'

#a = get_problem_list(url)
a = get_problem_list(url_college)

