#!/usr/bin/env python
# -*- coding:utf-8 -*-

import hashlib
from mytoolbox import mkdir_if_missing, textdump, textread
import requests
import os

def get_hash(targ_str):
    return hashlib.md5(targ_str).hexdigest()

def get_html(url):
    target_html = requests.get(url).text
    return target_html

def cache_webpage(url, base_dir=None, again=False):
    if base_dir is None:
        base_dir = '.webtoolcache'
        mkdir_if_missing(base_dir)
    targ_hash = get_hash(url)
    cache_path = os.path.join(base_dir, targ_hash)
    if os.path.exists(cache_path) and not again:
        return textread(cache_path)
    html = get_html(url)
    textdump(cache_path, [html])
    return html
