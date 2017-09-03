#!/usr/bin/env python
# -*- coding:utf-8 -*-

from bottle import route, run, template, post, static_file, request
from mytoolbox import *
from db_toolbox import *
from settings import *
import os
import pprint
set_debugger(send_email=False, error_func=None)

@route('/static/<filename>')
def get_static_file(filename):
    return static_file(filename, root=DATA_DIR)

@route('/files/<filename>')
def get_jscss_file(filename):
    return static_file(filename, root=FILE_DIR)

@route('/')
def index():
    return template('index.html',
                    server=HOST_NAME,
                    port=PORT)

@post('/save')
def save():
    keys = request.forms.keys()
    values = [request.forms.get(cur_key) for cur_key in keys]

    data_dict = {key: value for key, value in zip(keys, values)}

    update_db(data_dict)
    return

# request.forms.keys()

@post('/load')
def load():
    keys = request.forms.get('keys').split(',')
    results = get_from_db(keys)
    return results

@post('/delete')
def delete():
    raise NotImplementedError()

create_table()
run(host=HOST_NAME, port=PORT, debug=True, reloader=True)
