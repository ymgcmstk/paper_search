#!/usr/bin/env python
# -*- coding:utf-8 -*-

from settings import *

QUERY_SELECT_IN = 'SELECT title, authors, url FROM %s ' % TABLE_NAME + \
               'WHERE lower(title) in (%s)'
QUERY_SELECT_LIKE = 'SELECT title, authors, url FROM %s ' % TABLE_NAME + \
                    'WHERE lower(title) LIKE %s'
QUERY_REPLACE = 'INSERT OR REPLACE INTO %s ' % TABLE_NAME + \
                '(title, authors, url, conference) VALUES %s'

def create_table():
    CURSOR.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='%s'" % (TABLE_NAME))
    temp = CURSOR.fetchone()
    if temp[0] > 0:
        return

    query = "CREATE TABLE %s(%s)" % (TABLE_NAME, ','.join(PAPERS_VARS))
    query = query.replace('AUTO_INCREMENT', 'AUTOINCREMENT').replace('INT(11)', 'INTEGER')
    CURSOR.execute(query)
    CONNECTOR.commit()

def get_from_db(keys):
    # query = QUERY_SELECT_IN % ','.join(['"%s"' % cur_key for cur_key in keys])
    # query = QUERY_SELECT_MATCH % ' '.join([cur_key for cur_key in keys])
    names = ['title', 'authors', 'conference', 'url']
    query = 'SELECT %s FROM %s WHERE ' % (','.join(names), TABLE_NAME)

    for key in keys:
        query += "lower(title) LIKE '%%%s%%' AND " % key
    query = query[:-5]
    CURSOR.execute(query)
    results = CURSOR.fetchall()
    mod_results = []
    for cur_res in results:
        cur_data = {name: res for name, res in zip(names, cur_res)}
        mod_results.append(cur_data)
    # results = {i[0]: i[1] for i in CURSOR.fetchall()}
    return mod_results

def get_like_from_db(key):
    query = QUERY_SELECT_LIKE % key
    CURSOR.execute(query)
    results = {i[0]: i[1] for i in CURSOR.fetchall()}
    return results

def update_db(all_data):
    query = QUERY_REPLACE % ', '.join(['("%s", "%s", "%s", "%s")' % (data['title'], data['authors'], data['url'], data['conference']) for data in all_data])
    CURSOR.execute(query)
    CONNECTOR.commit()
