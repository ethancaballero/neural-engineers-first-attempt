# -*- coding: utf-8 -*-
from sys import argv
import time
import argparse
import shutil
import os
import re
import requests
import urllib2
from pprint import pprint
from bs4 import BeautifulSoup
import html2text
import concurrent.futures
import time
import subprocess


#p = subprocess.Popen(['pgrep phantomjs | xargs kill'], stdout=subprocess.PIPE, shell=True)

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'



#browser=webdriver.PhantomJS()
"""
'''THIS IS CODE TO LOGIN'''
def login(browser):
    browser.set_window_size(1124, 850)

    print 'browser'
    print browser

    url_home_page = 'https://www.hackerearth.com/challenges/'

    url2 = 'https://www.wikipedia.org/'

    browser.get(url_home_page)                                                    # This opens a firefox console  
    browser.implicitly_wait(1)
    #browser.implicitly_wait(.5)

    time.sleep(2)
    #time.sleep(.5)

    login_but=browser.find_element_by_xpath("//li[contains(@class,'nav-bar-menu login-menu-btn')]")

    webdriver.ActionChains(browser).click(login_but).perform()

    username = browser.find_element_by_id("id_login")
    password = browser.find_element_by_id("id_password")

    username.send_keys("gesturefm@gmail.com")
    password.send_keys("turnt123")

    browser.find_element_by_name("submit").click()
    browser.implicitly_wait(1)
    #browser.implicitly_wait(.5)
    time.sleep(1)
    #time.sleep(.5)

    return 0.0

'''LOGIN END'''
"""
"""
#'''
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    future_to_url = {executor.submit(login, browsers[i]): i for i in xrange(len(browsers))}
    for future in concurrent.futures.as_completed(future_to_url):
        data = future.result(timeout=1)
        #'''

#login(browsers[0])
"""
"""
def get_solution_ids(i, l):
    browser = browsers[0]
    browser.set_window_size(1124, 850)
    #browser=webdriver.Chrome()

    print i
    #jhgfhjg

    #url="https://www.hackerearth.com/problem/algorithm/assorted-arrangement-3/activity/"
    url="https://www.hackerearth.com/problem/algorithm/" + i + "/activity/"

    print url


    browser.get(url)                                                    # This opens a firefox console  
    browser.implicitly_wait(20)

   
    end=browser.find_element_by_xpath("//p[contains(@class,'no-scroll-content-message body-font gray-text hidden align-center') and .//text()='No more submissions to show.']")

    viewbut=browser.find_element_by_xpath("//a[contains(@class,'load-scroll-content button btn-blue hidden') and .//text()='View More']")
    dots=browser.find_element_by_xpath("//div[contains(@class,'scroll-content-loader dots-loader')]")

    
    '''
    end.is_displayed()
    while not end.is_displayed():
        for _ in xrange(0, 1000):
            if end.is_displayed():
                break
            webdriver.ActionChains(browser).move_to_element(viewbut).click(viewbut).perform()
        #break    
            '''

    i = 0
    #while not end.is_displayed():
    #while (not end.is_displayed()) and (not i > 1500):
    #while (not end.is_displayed()) and (not i > 3000):
    #while (not end.is_displayed()) and (not i > 10000):
    start = time.time()
    #while (not end.is_displayed()) and (not i > 5000):
    while (not end.is_displayed()) and (not i > 10000):
        i+=1
        if i % 50 == 0:
            print i
        webdriver.ActionChains(browser).move_to_element(viewbut).click(viewbut).perform()
    runtime = time.time() - start
    print runtime



    submission_ids = browser.find_elements_by_xpath('//*[contains(@id,"submission-row")]/td[6]/a')
    user_ids = browser.find_elements_by_xpath('//*[contains(@id,"submission-row")]/td[1]/a')
    #results = browser.find_elements_by_xpath('//*[contains(@id,"submission-row")]/td[2]/span')
    results = browser.find_elements_by_xpath('//*[contains(@id,"submission-row")]/td[2]/span[1]')
    language = browser.find_elements_by_xpath('//*[contains(@id,"submission-row")]/td[5]')

    submission_data = []

    submission_ids_c = []
    submission_ids_p = []

    print len(submission_ids), len(user_ids), len(results), len(language)

    for idx, i in enumerate(submission_ids):
        print submission_ids[idx].get_attribute("href"), user_ids[idx].get_attribute("href"), results[idx].get_attribute("class"), language[idx].text

    '''
    for i in submission_ids:
        print i.get_attribute("href")

    for i in user_ids:
        print i.get_attribute("href")

    for i in results:
        print i.get_attribute("class")

    for i in language:
        print i.text
        #'''

    print 'len(submission_ids)'
    print len(submission_ids)

    for i in range(len(submission_ids)):
        if results[i].get_attribute("class") == 'fa-green':
            if user_ids[i].get_attribute("href") != user_ids[i-1].get_attribute("href"):
                if language[i].text == 'C++':
                    #submission_data.append([submission_ids[i].get_attribute("href"), user_ids[i].get_attribute("href"), results[i].get_attribute("class"), language[i].text])
                    submission_ids_c.append(submission_ids[i].get_attribute("href"))
                elif language[i].text == 'Python':
                    #submission_data.append([submission_ids[i].get_attribute("href"), user_ids[i].get_attribute("href"), results[i].get_attribute("class"), language[i].text])
                    submission_ids_p.append(submission_ids[i].get_attribute("href"))

    print 'submission_ids_c'
    print len(submission_ids_c)
    print submission_ids_c

    print 'submission_ids_p'
    print len(submission_ids_p)
    print submission_ids_p

    return submission_ids_p, submission_ids_c
    """

def get_solution_ids(i, l):

    #print i

    url="https://www.hackerearth.com/problem/algorithm/" + i + "/activity/"
    t = requests.get(url)
    tmp_string = t.headers["set-cookie"]
    csrf_token = re.findall(r"csrftoken=\w*", tmp_string)[0][10:]
    problem_id = re.findall(r"/AJAX/submissions/problem/algorithm/(.*?)/',", t.text)
    #print problem_id

    response = {}
    response["host"] = "www.hackerearth.com"
    response["user-agent"] = user_agent
    response["accept"] = "application/json, text/javascript, */*; q=0.01"
    response["accept-language"] = "en-US,en;q=0.5"
    response["accept-encoding"] = "gzip, deflate"
    response["content-type"] = "application/x-www-form-urlencoded"
    response["X-CSRFToken"] = csrf_token
    response["X-Requested-With"] = "XMLHttpRequest"
    #response["Referer"] = "https://www.hackerearth.com/submissions/" + handle + "/"
    response["Referer"] = url
    response["Connection"] = "keep-alive"
    response["Pragma"] = "no-cache"
    response["Cache-Control"] = "no-cache"
    response["Cookie"] = tmp_string

    #url = []
    pages = []
    #for i in xrange(1, 6):
    #for i in xrange(1, 10):
    #for i in xrange(1, 20):
    for i in xrange(1, 25):
        if l == 'python':
            url = "https://www.hackerearth.com/AJAX/submissions/problem/algorithm/" + str(problem_id[0]) + "/?result=AC&lang=Python&page=" + str(i)
            page = requests.get(url, headers=response)
            if str(page) == "<Response [503]>":
                while str(page) == "<Response [503]>":
                    time.sleep(1)
                    page = requests.get(url, headers=response)
            pages.append(page)
        elif l == 'c++':
            url = "https://www.hackerearth.com/AJAX/submissions/problem/algorithm/" + str(problem_id[0]) + "/?result=AC&lang=C%2B%2B&page=" + str(i)
            page = requests.get(url, headers=response)
            if str(page) == "<Response [503]>":
                while str(page) == "<Response [503]>":
                    time.sleep(1)
                    page = requests.get(url, headers=response)
            pages.append(page)
        else:
            pass
        print i

    #print len(pages)

    trs = []
    #body = ''
    #'''
    for i in pages:
        if len(i.text) > 15:
            body = i.json()["data"]
            #print len(body)
            soup = BeautifulSoup(body, "lxml")
            trsub = soup.find("tbody").find_all("tr")
            for i in trsub:
                trs.append(i)
                #'''

    '''
    body = ''
    for i in pages:
        if len(i.text) > 15:
            body += i.json()["data"]
            #print len(body)
    soup = BeautifulSoup(body, "lxml")
    trs = soup.find("tbody").find_all("tr")
    '''


    #print trs

    #body += i.json()["data"]


    #body = page1.json()["data"] + page2.json()["data"]
    #json_response["data"]


    #body = json_response["data"]
    '''
    soup = BeautifulSoup(body, "lxml")
    trs = soup.find("tbody").find_all("tr")    
    '''


    #print trs

    links = []

    #print len(trs)

    for tr in trs:
        partial = False
        for i in tr.find_all('i', class_=True):
            #print str(i)
            if "fa-orange" in str(i):
                #print "partial"
                partial = True

        if partial == False:
            for a in tr.find_all('a', class_='link-13'):
                #print a
                link = re.search('"/submission/(.*?)/"', str(a)).group(1)
                #print "link"
                #print link
            if len(links) < 50:
                links.append(link)
            else:
                break



        #all_tds = tr.find_all("td")
        #all_as = tr.find_all("a")
        #print tr
        #print all_tds


    print len(links)
    #print links

    #fdghfdh

    return links

'''
def get_solutions(ids):
    solutions = []
    failed_to_download_s = []
    failed_soultion_list = []
    for idx, i in enumerate(ids):
        if len(solutions) < 50:
            solutions.append(get_solution(i))
            '''
"""
def get_solutions(solution_ids):
    solutions = {}
    failed_to_download_s = []
    #for i in solution_id

    print "len(solution_ids)"
    print len(solution_ids)
    '''
    for j in xrange(0, len(solution_ids), 50):
        print j
        print solution_ids[j:j + 50]
        #'''
    #'''
    for j in xrange(0, len(solution_ids), len(browsers)):
        #print j
        a = solution_ids[j:j + len(browsers)]
        #'''

        #adfasf
        worked = False
        #swhile not worked:
        try:
            '''BATCH SIZE OF TEN IS LIMIT WHEN YOU USE MAX_WORKERS = 50'''
            with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
            #with concurrent.futures.ProcessPoolExecutor(max_workers=200) as executor:
                #future_to_url = {executor.submit(get_solution, i): i for i in solution_ids}
                future_to_url = {executor.submit(get_solution, idx, i): (idx, i) for idx, i in enumerate(a)}
                for future in concurrent.futures.as_completed(future_to_url):
                    data = future.result()

                    print 'data'
                    print data

                    if data[2] == None and data[1] != None and len(solutions) < 50:
                        solutions[data[0]] = data[1]

                    worked = True

                    if len(solutions) >= 50:
                        worked = True
                        break



                sgfdsgfds

        except WebDriverException:
            print "exception_i"
            time.sleep(.01)

    print solutions

    #sgfdsgfds

    return solutions
    #"""

def get_solutions(solution_ids):
    sample_inputs, sample_outputs = get_samples(solution_ids[0])
    solutions = []
    for i in solution_ids:
        solutions.append(get_solution(i))

    '''
    print 'sample_inputs'
    print sample_inputs
    print 'sample_outputs'
    print sample_outputs
    print 'solutions'
    print solutions
    '''

    #sgfds

    return sample_inputs, sample_outputs, solutions


#'''
#def get_solution(idx, i):
def get_solution(i):
    #url = "https://www.hackerearth.com/submission/5159653/"
    url = "https://www.hackerearth.com/submission/" + i
    cookies = {}
 
    '''YOU NEED TO RELOOK UP THIS VALUE AND UPDATE IT REGULARLY'''
    cookies["lordoftherings"]="13c082ac336859d586aa5364c086d26f:fdf0d9623bc1bfe6727dc755f6ceb2e8"

    t = requests.get(url, cookies=cookies)
    if str(t) == "<Response [503]>":
        while str(t) == "<Response [503]>":
            time.sleep(1)
            t = requests.get(url, cookies=cookies)

    solution_url_clones = re.findall('submission/key/(.+?)/', t.text)
    solution_key = solution_url_clones[0]

    solution_url = "https://www.hackerearth.com/submission/key/" + solution_key

    solution_request = requests.get(solution_url)
    if str(solution_request) == "<Response [503]>":
        while str(solution_request) == "<Response [503]>":
            time.sleep(1)
            solution_request = requests.get(solution_url)

    #print solution_request.text

    html_content = solution_request.text

    soup = BeautifulSoup(html_content)

    text = soup.select("pre")

    #print 'text'
    #print text

    failed_to_download_s = []
    solution = ''
    if len(text)==0:
        failed_to_download_s.append(i)
    else:
        body = BeautifulSoup(str(text[0]), "html.parser").get_text()

        body = body.replace("\\","\\\\").encode('utf-8').decode('string-escape')
        #solutions.append([i, body.encode('utf-8').decode('string-escape')])
        #print "body"
        #print body
        solution = body

    #return solutions, failed_to_download_s

    #return solution

    return i, solution, failed_to_download_s
    #'''

def get_samples(i):
    #url = "https://www.hackerearth.com/response/submission/5159653/"
    url = "https://www.hackerearth.com/response/submission/" + i + "/"
    #url = "https://www.hackerearth.com/submission/" + i
    cookies = {}

    '''YOU NEED TO RELOOK UP THIS VALUE AND UPDATE IT REGULARLY'''
    cookies["lordoftherings"]="13c082ac336859d586aa5364c086d26f:fdf0d9623bc1bfe6727dc755f6ceb2e8"

    t = requests.get(url, cookies=cookies)
    if str(t) == "<Response [503]>":
        while str(t) == "<Response [503]>":
            time.sleep(1)
            t = requests.get(url, cookies=cookies)

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

    url += "/AJAX/"
    #url = "https://www.hackerearth.com/response/submission/5159653/AJAX/"


    tmp = requests.get(url, headers=response)
    if str(tmp) == "<Response [503]>":
        while str(tmp) == "<Response [503]>":
            time.sleep(1)
            tmp = requests.get(url, headers=response)

    #sgfdsgd

    test_urls = re.findall('<a href="(.+?)\\?Signature=', tmp.text)
    #print test_urls
    input_urls=[]
    output_urls=[]
    for i in range(len(test_urls)):
        if i % 3 == 0:
            input_urls.append(test_urls[i])
        elif i % 3 == 2:
            output_urls.append(test_urls[i])
        else:
            pass

    inputs = []
    outputs = []
    for j in input_urls:
        input_request = requests.get(j)
        if str(input_request) == "<Response [503]>":
            while str(input_request) == "<Response [503]>":
                time.sleep(1)
                input_request = requests.get(j)
        inputs.append(input_request.text)

    for j in output_urls:
        output_request = requests.get(j)
        if str(output_request) == "<Response [503]>":
            while str(output_request) == "<Response [503]>":
                time.sleep(1)
                output_request = requests.get(j)
        outputs.append(output_request.text)

    '''NEED SEARCH TO GRAB ACTUAL SOLUTION URL AS WELL'''
    '''ONLY ONE PERSON'S I/O CHALLENGES WILL BE GRABBED'''

    solution_url = re.findall('submission/key/(.+?)/', t.text)

    return inputs, outputs
    #'''


def get_tags():
    '''
    <tr id="row">
    <a href="/problem/algorithm/mancunian-and-nancy-play-a-game-1/" class="link-13 track-problem-link">
    <td class="medium-col dark track-tags">
                    Graph Theory, Medium
                </td>
    '''
    pass


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
        body = re.search('/submit/(.*)" t', raw)

        if body != None:
            w = body.group(1)
            message = str(w)
            #if message != 'easy' and message != 'medium' and message != 'hard' and message != 'challenge' and message != 'extcontest' and message != 'school':
            messages.append(message)

    return messages


def get_descriptions(problem):
    descriptions = []
    left_out = []
    failed_to_download_d = []
    #print problem_list
    #for i in problem_list:

    url = 'https://www.hackerearth.com/problem/algorithm/' + problem

    #print url

    page = requests.get(url)

    if str(page) == "<Response [503]>":
        while str(page) == "<Response [503]>":
            time.sleep(1)
            page = requests.get(url)

    html_content_all = page.text

    if re.search('"message":"requests limit exhausted"', html_content_all) != None:
        while re.search('message":"requests limit exhausted', html_content_all) != None:
            time.sleep(1)
            page = requests.get(url)
            html_content_all = page.text

    if html_content_all==None:
        failed_to_download_d.append(i)

    soup = BeautifulSoup(html_content_all)

    html_content_1 = soup.findAll("div", { "class" : "starwars-lab" })
    html_content_2 = soup.findAll("div", { "class" : "less-margin-2 input-output-container" })
    html_content_3 = soup.findAll("div", { "class" : "standard-margin" })

    #raw = BeautifulSoup(str(html_content[0]).replace("</p>", "\n</p>").replace("<sup>", "<sup>^").replace("\le", u"≤").replace("\ge", u"≥").replace("\lt", "<").replace("\gt", ">"), "html.parser").get_text()
    raw = BeautifulSoup(str(html_content_1[0]).replace("</p>", "\n</p>").replace("<sup>", "<sup>^"), "html.parser").get_text() + BeautifulSoup(str(html_content_2[0]).replace("</p>", "\n</p>").replace("<sup>", "<sup>^"), "html.parser").get_text() + BeautifulSoup(str(html_content_3[0]).replace("</p>", "\n</p>").replace("<sup>", "<sup>^"), "html.parser").get_text()

    #if re.search("https://d320jcjashajb2.cloudfront.net/media/uploads", str(html_content_all)) == None and re.search('"message":"Problem is not visible now. Please try again later."', str(html_content_all)) == None and re.search('Statement is not available', str(html_content_all)) == None:
    if re.search("https://d320jcjashajb2.cloudfront.net/media/uploads", html_content_all) == None and re.search('"message":"Problem is not visible now. Please try again later."', html_content_all) == None and re.search('Statement is not available', html_content_all) == None:
        raw = raw.replace("\n\n\n\n\n\n", "")
        #raw = raw.replace("\n\n\n\n\n", "\n")

        raw = raw.replace("\n\n\n", "\n")
        raw = raw.replace("\n\n\n", "\n\n")

        raw = raw.replace("\n\n\n", "\n\n")

        raw = raw.replace("<sup>", "<sup>^")

        raw = raw.replace("\in", u"∈").replace('$$', '')

        raw = raw.replace(" <=", u" ≤").replace(" >=", u" ≥").replace("<=", u" ≤ ").replace(">=", u" ≥ ").replace(u"≤  ", u"≤ ").replace(u"≥  ", u"≥ ").replace("\le", u"≤").replace("\ge", u"≥").replace("\lt", "<").replace("\gt", ">")

        raw = re.sub('Subtasks(.+?)SAMPLE INPUT', 'SAMPLE INPUT', raw, flags=re.S)

        raw = re.sub('Time Limit:(.+)', '', raw, flags=re.S)

        raw = re.sub('See Russian translation\n\n', '', raw, flags=re.S)
        raw = re.sub('See Russian translation', '', raw, flags=re.S)

        raw = raw.replace("\\","\\\\")

        descriptions.append(raw.encode('utf-8').decode('string-escape'))

    else:
        #left_out.append(i)
        #descriptions.append(raw.encode('utf-8').decode('string-escape'))
        left_out.append(problem)

        #hjgf

    #print 'descriptions'    
    #print descriptions[0]

    #asasdf

    return descriptions, left_out, failed_to_download_d


def download_all_challenge_names(filename):
    target = open(filename, 'w')

    problems = []
    
    target.write(str(problems))

    #'''
    
    #'''

#download_all_challenge_names('codechef_problem_names.txt')

def download_descriptions_solutions(filename, index_n):
    #root_dir = 'hackerearth_data'
    root_dir = 'hackerearth_data_working'

    if filename == 'problems.txt':
        category = "problems_normal"
    elif filename == 'problems_college.txt':
        category = "problems_college"

    file = open(filename, 'r')
    f = open(filename, 'r')

    index_n_int = int(index_n)

    start = index_n_int + (500*index_n_int)
    end = start + 499

    all_names = []

    for line in f:
        raw = eval(str(line))

    #print raw

    a = ""
    b = ""

    all_names = raw

 
    #language = ["python", "c++"]
    language = ["c++", "python"]

    #print "wtf"

    #print descriptions, left_out, failed_to_download_d

    for idx, i in enumerate(all_names):
        #print i
        #descriptions, left_out, failed_to_download_d = get_descriptions(i)
        '''IT'S SET TO THE SAME ONE EVERY TIME RIGHT HERE; NEED TO REMOVE'''
        #i = "sort-the-array-5"
        #i = "bits-transformation-1"
        #i = "string-division"
        #i = "little-achraf-in-who-wants-to-be-a-millionaire"
        #i = "terrible-chandu"
        description, left_out, failed_to_download_d = get_descriptions(i)
        #print i
        if i not in left_out:
            print i

            if not os.path.exists(root_dir):
                os.makedirs(root_dir)

            #'''
            cat_dir = root_dir + "/" + category

            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)

            save_dir = cat_dir + "/" + i
            #'''

            #save_dir = root_dir + "/" + i[0] + "_" + i[1]

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            description_dir = save_dir + "/description"

            if not os.path.exists(description_dir):
                os.makedirs(description_dir)

            #print description
            #print description[0]
            #print 'description'

            description_file_path = description_dir + "/description.txt"
            description_file = open(description_file_path, 'w')
            description_file.write(description[0])

            #asdf

            ids = []

            ids_l = []

            #ids_p, ids_c = get_solution_ids(i, language)

            for l in language:
                ids = get_solution_ids(i, l)
                ids_l.append(ids)

                #ids_p, ids_c = get_solution_ids(i, language)

                print l

                '''
                if l == 'python':
                    ids = ids_p
                elif l == 'c++':
                    ids = ids_c
                    '''

                print ids
                #solutions, failed_to_download_s = get_solutions(i, ids)
                #solutions, failed_to_download_s = get_solutions(ids)
                if len(ids) > 0:
                    sample_inputs, sample_outputs, solutions = get_solutions(ids)
                #print failed_to_download_s

                solution_dir = save_dir + "/solutions_" + l

                if not os.path.exists(solution_dir):
                    os.makedirs(solution_dir)

                sample_dir = save_dir + "/samples"

                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                print "solutions"
                print len(solutions)
                #print solutions

                for jdx, j in enumerate(sample_inputs):
                    #print j
                    if len(sample_inputs[jdx]) < 10000:
                        #j_num = re.findall('submission/(.+?)/', str(j))
                        solution_file_path = sample_dir + "/" + str(jdx+1) + "_input" + ".txt"
                        solution_file = open(solution_file_path, 'w')
                        solution_file.write(sample_inputs[jdx])

                for jdx, j in enumerate(sample_outputs):
                    #print j
                    if len(sample_outputs[jdx]) < 10000:
                        #j_num = re.findall('submission/(.+?)/', str(j))
                        solution_file_path = sample_dir + "/" + str(jdx+1) + "_output" + ".txt"
                        solution_file = open(solution_file_path, 'w')
                        solution_file.write(sample_outputs[jdx])

                for jdx, j in enumerate(solutions):
                    #print j
                    if len(j[1]) < 10000:
                        '''probably need to change this line'''
                        #j_num = re.findall('submission/(.+?)/', str(j))
                        solution_file_path = solution_dir + "/" + str(j[0]) + ".txt"
                        solution_file = open(solution_file_path, 'w')
                        solution_file.write(j[1])

                    '''
                    print j
                    fdghfd
                    #j_num = re.findall('submission/(.+?)/', str(j))
                    j_num = re.findall('submission/(.+?)/', str(j[0]))
                    #solution_file_path = solution_dir + "/" + j[0] + ".txt"
                    #solution_file_path = solution_dir + "/" + j_num[0] + ".txt"
                    solution_file_path = solution_dir + "/" + str(j_num[0]) + ".txt"
                    print solution_file_path
                    solution_file = open(solution_file_path, 'w')
                    solution_file.write(j[1])
                    '''

            #remove problems with zero solutions
            if len(ids_l[0]) == 0 and len(ids_l[1]) == 0:
            #if len(ids_p) == 0 and len(ids_c) == 0:
                shutil.rmtree(save_dir)

        #sasdsdfasd

        #url = 'https://www.codechef.com/status/%d?sort_by=All&sorting_order=asc&language=4&status=15&handle=&Submit=GO' % (name)

'''
print "Finished download process"
if len(failed_to_download) > 0:
    print "Following challenges failed to download: " + str(failed_to_download)
    '''


#download_all_challenge_names('codechef_problem_names.txt') 
parser = argparse.ArgumentParser()
parser.add_argument('--index', type=str, default="index", help='')
args = parser.parse_args()

index_n = args.index

#download_descriptions_solutions('challenges.txt', index_n)

'''ADD MODULE TO GRAB DESCRIPTIONS WITH/WITHOUT ANNOTATIONS FOR TOKENS TO BE CONVERTED TO VARIABLES'''
#'''
download_descriptions_solutions('problems.txt', index_n)
#'''
#'''
#get_solutions(ids)
'''
get_solutions('https://www.hackerearth.com/submission/4634147/')
#'''

#browser.close()
#browser.quit()

#download_descriptions_solutions('challenges_1.txt', index_n)

#get_descriptions(['interval-count-12', 'final-battle', 'benny-and-universal-numbers'])

#download_descriptions_solutions('codechef_problem_names_easy.txt')
#download_descriptions_solutions('codechef_problem_names_easy_short.txt')

"""

    
print get_solution('5159653')
a = get_samples('5159653')
print a
print a[0]
print a[0][0]
#"""






