#!/usr/bin/env python
# -*- coding:utf-8 -*-

from settings import *

QUERY_SELECT = 'SELECT key, value FROM %s ' % TABLE_NAME + \
               'WHERE key in (%s)'
QUERY_REPLACE = 'INSERT OR REPLACE INTO %s ' % TABLE_NAME + \
                '(key, value) VALUES %s'
# QUERY_REPLACE % ', '.join(['(%s,%s)' % (key, value) for key, value in data_dict])

def create_table():
    CURSOR.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='%s'" % (TABLE_NAME))
    temp = CURSOR.fetchone()
    if temp[0] > 0:
        return

    query = "CREATE TABLE %s(%s)" % (TABLE_NAME, ','.join(KVS_VARS))
    query = query.replace('AUTO_INCREMENT', 'AUTOINCREMENT').replace('INT(11)', 'INTEGER')
    CURSOR.execute(query)
    CONNECTOR.commit()

def get_from_db(keys):
    print QUERY_SELECT % ','.join(keys)
    CURSOR.execute(QUERY_SELECT % ','.join(keys))
    results = {i[0]: i[1] for i in CURSOR.fetchall()}
    return results

def update_db(data_dict):
    query = QUERY_REPLACE % ', '.join(['("%s", "%s")' % (key, value) for key, value in data_dict.iteritems()])
    print query
    CURSOR.execute(query)
