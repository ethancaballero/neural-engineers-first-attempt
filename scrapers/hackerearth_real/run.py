import subprocess

def main():
    for i in xrange(0, 6):
        #print i
        print 'python hackerearth_whole_scrape_split.py --index %s' % (i)
        p = subprocess.Popen(['python hackerearth_whole_scrape_split.py --index %s' % (i)], stdout=subprocess.PIPE, shell=True)
        #output_sentiment, err = p.communicate()

if __name__ == '__main__':
    main()