#!/usr/bin/env python
# -*- coding:utf-8 -*-

from bottle import route, run, template, post, static_file, request
from db_toolbox import *
from settings import *

def is_ml(conf):
    for cur_conf in ['nips', 'icml']:
        if cur_conf in conf.lower():
            return True
    return False

@route('/static/<filename>')
def get_static_file(filename):
    return static_file(filename, root=DATA_DIR)

@route('/files/<filename>')
def get_jscss_file(filename):
    return static_file(filename, root=FILE_DIR)

@route('/search')
def index():
    papers = []
    try:
        keys = request.query['keys'] # request.forms.get('keys')
    except:
        keys = None
    try:
        ml_flg = request.query['ml'] == 'ml' # request.forms.get('keys')
    except:
        ml_flg = False

    if keys is not None:
        if len(keys) > 0:
            papers = get_from_db(keys.split())
            papers = sorted(papers, key=lambda x: x['conference'][::-1])[::-1]
            if ml_flg:
                papers = [cur_p for cur_p in papers if is_ml(cur_p['conference'])]
    return template('index.html',
                    server=HOST_NAME,
                    port=PORT,
                    papers=papers,
                    keys=keys)

@post('/load')
def load():
    keys = request.forms.get('keys').split()
    print(keys)
    results = get_from_db(keys)
    return results

run(host=HOST_NAME, port=PORT, debug=True, reloader=True)
