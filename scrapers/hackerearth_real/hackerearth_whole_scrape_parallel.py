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


p = subprocess.Popen(['pgrep phantomjs | xargs kill'], stdout=subprocess.PIPE, shell=True)


selenium_present = True

try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
  
except ImportError:
    selenium_present = False
if not selenium_present:
    print '''Please install python package "selenium" to use HEtoGit'''
    exit(0)

if len(argv) < 2:
    print 'Usage: hetogit.py <username>'
    exit(0)
else:
    username = argv[1]

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'


#browser=webdriver.Firefox()
print "b"
dcap = dict()
dcap["phantomjs.page.settings.userAgent"] = (user_agent)

#browser = webdriver.PhantomJS(desired_capabilities = dcap)
browsers = {}
#for i in xrange(0,20):
for i in xrange(0,2):
    browsers[i] = webdriver.Chrome()
    print i

print browsers

#browser=webdriver.PhantomJS()

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
#'''
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    future_to_url = {executor.submit(login, browsers[i]): i for i in xrange(len(browsers))}
    for future in concurrent.futures.as_completed(future_to_url):
        data = future.result(timeout=1)
        #'''

#login(browsers[0])


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

    """ 
    Traverse till "No more submission" element is not visible and when
    you scroll for 5 times you will see "View More" button so you need
    to click it to load more data
    """

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
'''
def get_solutions(ids):
    solutions = []
    failed_to_download_s = []
    failed_soultion_list = []
    for idx, i in enumerate(ids):
        if len(solutions) < 50:
            solutions.append(get_solution(i))
            '''

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

#'''
def get_solution(idx, i):
    browser = browsers[idx]
    solutions = []
    failed_to_download = []
    failed_solution_list = []

    failed_to_download = None
    solution = None

    url = str(i)

    print url
    print idx


    #open tab
    #browser.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 't') 
    #browser.execute_script("window.open('','_blank');")
    #browser.switch_to.window(browser.window_handles[idx+1])

    browser.get(url)                                                   # This opens a firefox console  

    throw_away = WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "fa-file-code-o"))
            )

    results = browser.find_elements_by_xpath('//*[contains(@class,"result-icon")]')

    failed_solution = 0
    for r in results:
        #print 'r.get_attribute("class")'
        #print r.get_attribute("class")
        if r.get_attribute("class") != 'fa fa-check-circle fa-green result-icon tool-tip':
            failed_solution = 1

    print failed_solution

    if failed_solution == 1:
        failed_solution_list.append(i)

    if failed_solution != 1:
        link = browser.find_element_by_xpath('//*[contains(@src,"/submission/key/")]')

        #print 'link.get_attribute("src")'
        #print link
        #print link.get_attribute("src")

        solution_link = link.get_attribute("src")
        
        print solution_link

        page = requests.get(solution_link)
        if str(page) == "<Response [503]>":
            while str(page) == "<Response [503]>":
                time.sleep(1)
                page = requests.get(solution_link)
        html_content = page.text

        #print html_content

        #dsgfd

        soup = BeautifulSoup(html_content)

        text = soup.select("pre")
        #print text

        #print BeautifulSoup(str(text[0]), "html.parser").get_text()
        #print BeautifulSoup(str(text[0]), "html.parser").get_text().encode('utf-8').decode('string-escape')

        #dsgfd

        #failed_to_download = None
        #solution = None

        if len(text)==0:
            failed_to_download = i
        else:
            body = BeautifulSoup(str(text[0]), "html.parser").get_text()

            body = body.replace("\\","\\\\").encode('utf-8').decode('string-escape')
            #solutions.append([i, body.encode('utf-8').decode('string-escape')])
            #solution = [i, body]
            print "body"
            print body
            solution = body


            """
            if body != solutions[-1][1]:
                solutions.append([i, body])
                """

    #close the tab
    #browser.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 'w')
    #browser.execute_script("window.close('','_blank');")

    #return solution, failed_to_download_s
    #return solution_id, solution, failed_to_download
    return i, solution, failed_to_download
    #'''

'''
def get_solution(i):
    solutions = []
    failed_to_download = []
    failed_soultion_list = []

    failed_to_download = None
    solution = None

    url = str(i)

    print url

    #open tab
    browser.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 't') 

    #worked = False

    #while not worked:
    #try:
    browser.get(url)                                                   # This opens a firefox console  

    print "1"

    browser.implicitly_wait(.5)
    time.sleep(.5)

    results = browser.find_elements_by_xpath('//*[contains(@class,"result-icon tool-tip")]')

    print "2"

    failed_solution = 0
    for r in results:
        if r.get_attribute("class") != 'fa fa-check-circle fa-green result-icon tool-tip':
            failed_solution = 1

    print "3"

    if failed_solution == 1:
        failed_soultion_list.append(i)

    print "4"

    if failed_solution != 1:
        link = browser.find_element_by_xpath('//*[contains(@src,"/submission/key/")]')

        print "5"

        solution_link = link.get_attribute("src")

        print "6"
        
        print solution_link

        page = requests.get(solution_link)

        print "7"

        if str(page) == "<Response [503]>":
            while str(page) == "<Response [503]>":
                time.sleep(1)
                page = requests.get(solution_link)
        html_content = page.text

        soup = BeautifulSoup(html_content)

        text = soup.select("pre")

        if len(text)==0:
            failed_to_download = i
        else:
            body = BeautifulSoup(str(text[0]), "html.parser").get_text()

            body = body.replace("\\","\\\\").encode('utf-8').decode('string-escape')

            solution = body

        #worked = True

    #except WebDriverException:
        #print "exception_i"
        #time.sleep(.01)


    browser.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 'w')

    return i, solution, failed_to_download
    '''



def get_tags():
    """
    <tr id="row">
    <a href="/problem/algorithm/mancunian-and-nancy-play-a-game-1/" class="link-13 track-problem-link">
    <td class="medium-col dark track-tags">
                    Graph Theory, Medium
                </td>
    """
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

    print url

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

    print "wtf"

    #print descriptions, left_out, failed_to_download_d

    for idx, i in enumerate(all_names):
        print i
        #descriptions, left_out, failed_to_download_d = get_descriptions(i)
        '''IT'S SET TO THE SAME ONE EVERY TIME RIGHT HERE; NEED TO REMOVE'''
        i = "sort-the-array-5"
        #i = "bits-transformation-1"
        #i = "string-division"
        #i = "little-achraf-in-who-wants-to-be-a-millionaire"
        #i = "terrible-chandu"
        description, left_out, failed_to_download_d = get_descriptions(i)
        print i
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

            ids_p, ids_c = get_solution_ids(i, language)

            for l in language:
                #ids = get_solution_ids(i, l)
                #ids_l.append(ids)

                print l

                if l == 'python':
                    ids = ids_p
                elif l == 'c++':
                    ids = ids_c

                print ids
                #solutions, failed_to_download_s = get_solutions(i, ids)
                #solutions, failed_to_download_s = get_solutions(ids)
                solutions = get_solutions(ids)
                #print failed_to_download_s

                solution_dir = save_dir + "/solutions_" + l

                if not os.path.exists(solution_dir):
                    os.makedirs(solution_dir)

                print "solutions"
                print len(solutions)
                print solutions

                for jdx, j in enumerate(solutions):
                    print j
                    if len(solutions[j]) < 10000:
                        j_num = re.findall('submission/(.+?)/', str(j))
                        solution_file_path = solution_dir + "/" + str(j_num[0]) + ".txt"
                        solution_file = open(solution_file_path, 'w')
                        solution_file.write(solutions[j])

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
            #if len(ids_l[0]) == 0 and len(ids_l[1]) == 0:
            if len(ids_p) == 0 and len(ids_c) == 0:
                shutil.rmtree(save_dir)

        #sasdsdfasd

    browser.close()
    browser.quit()

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
#'''
download_descriptions_solutions('problems.txt', index_n)
#'''
#'''
#get_solutions(ids)
'''
get_solutions('https://www.hackerearth.com/submission/4634147/')
#'''

browser.close()
browser.quit()

#download_descriptions_solutions('challenges_1.txt', index_n)

#get_descriptions(['interval-count-12', 'final-battle', 'benny-and-universal-numbers'])

#download_descriptions_solutions('codechef_problem_names_easy.txt')
#download_descriptions_solutions('codechef_problem_names_easy_short.txt')







