#!/bin/sh

#echo pgrep phantomjs | xargs kill
#time pgrep phantomjs | xargs kill

pgrep phantomjs | xargs kill

THIS_DIR=$(cd $(dirname $0); pwd)

export PYTHONPATH=${THIS_DIR}
cd "$THIS_DIR"

#echo python hackerearth_whole_scrape_split.py --index 1
#time python hackerearth_whole_scrape_split.py --index 1

#python hackerearth_whole_scrape_split.py --index 1



python hackerearth_whole_scrape_split.py --index 0 & python hackerearth_whole_scrape_split.py --index 1 & python hackerearth_whole_scrape_split.py --index 2 & python hackerearth_whole_scrape_split.py --index 3 & python hackerearth_whole_scrape_split.py --index 4 & python hackerearth_whole_scrape_split.py --index 5 && fg
#python hackerearth_whole_scrape_split.py --index 0 & python hackerearth_whole_scrape_split.py --index 1
#python hackerearth_whole_scrape_split.py --index 0 &
#python hackerearth_whole_scrape_split.py --index 1 &
#python hackerearth_whole_scrape_split.py --index 2 &
#python hackerearth_whole_scrape_split.py --index 3 &
#python hackerearth_whole_scrape_split.py --index 4 &
#python hackerearth_whole_scrape_split.py --index 5
#&& fg

#(echo python hackerearth_whole_scrape_split.py --index 0; echo python hackerearth_whole_scrape_split.py --index 1; python hackerearth_whole_scrape_split.py --index 2; python hackerearth_whole_scrape_split.py --index 3; python hackerearth_whole_scrape_split.py --index 4; python hackerearth_whole_scrape_split.py --index 5) | parallel

