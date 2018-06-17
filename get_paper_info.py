#!/usr/bin/env python
# -*- coding:utf-8 -*-

from mytoolbox import *
from web_toolbox import *
from db_toolbox import *
from settings import *

DBLP_URL = 'http://dblp.uni-trier.de/db/conf/%s/%s%s.html'

def get_paper_info(name, year, footer=''):
    name = name.lower()
    year = str(year)
    url = DBLP_URL % (name, name, year + footer)
    html = cache_webpage(url)
    if isinstance(html, list):
        html = ' '.join(html)
    no_url_mode = False
    html_splitted = html.split('<p><b>view</b></p><ul><li><a href="')[2:]
    if len(html_splitted) == 0:
        html_splitted = html.split('no documents available')[2:]
        no_url_mode = True
    all_data = []
    for seg in html_splitted:
        dobcol = seg.find('"')
        if no_url_mode:
            url = ''
        else:
            url = seg[:dobcol]
        temp_authors = seg.split('<span itemprop="author" itemscope itemtype=')[1:]
        authors = []
        for temp_auth in temp_authors:
            authors.append(temp_auth.split('<span itemprop="name">')[1].split('<')[0])
        title = seg.split('class="title" itemprop="name">')[1].split('.')[0].split('<')[0]
        data = {
            'title': title,
            'url': url,
            'authors': ', '.join(authors),
            'conference': name.upper() + year
        }
        all_data.append(data)
    if len(all_data) == 0:
        print('%s has been skipped as no paper are found.' % (name.upper() + year + footer))
        return all_data
    update_db(all_data)
    return all_data

if __name__ == '__main__':
    set_debugger(send_email=False, error_func=None)
    create_table()
    for conf in ['nips', 'icml', 'iccv', 'cvpr']:
        for year in range(2010, 2018):
            get_paper_info(conf, year)
    for year in range(2010, 2018)[::2]:
        cur_ind = 1
        while True:
            cur_data = get_paper_info('eccv', year, '-%d' % cur_ind)
            if len(cur_data) == 0:
                break
            cur_ind += 1
