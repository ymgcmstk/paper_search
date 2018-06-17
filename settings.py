#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sqlite3

HOST_NAME = '0.0.0.0'
PORT = 8085
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), __file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
FILE_DIR = os.path.join(ROOT_DIR, 'files')

# database name and table name
DB_NAME = 'papers'
TABLE_NAME = 'papers'
DB_FILE_NAME = os.path.join(DATA_DIR, '%s.db' % DB_NAME)

# connector and cursor
CONNECTOR = sqlite3.connect(DB_FILE_NAME)
CURSOR = CONNECTOR.cursor()

PAPERS_VARS = [
    'title TEXT PRIMARY KEY',
    'authors TEXT',
    'url TEXT',
    'conference TEXT',
]
