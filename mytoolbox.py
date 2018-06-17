#!/usr/bin/env python
# -*- coding:utf-8 -*-

import atexit
from collections import defaultdict
import commands
from contextlib import closing
import copy
import cPickle
import datetime
import httplib
import inspect
from IPython.core import ultratb
from IPython import embed
import json
import os
import pickle
import random
import scipy.io as sio
from scipy.stats import pearsonr
import select
import shutil
import socket
import subprocess
import sys
import time
import threading
import urllib
import urllib2

try:
    import ymgc_toolbox
    cp2www               = ymgc_toolbox.cp2www
    get_example          = ymgc_toolbox.get_example
    cPicklecache         = ymgc_toolbox.cPicklecache
    load_metainfo        = ymgc_toolbox.load_metainfo
    save_metainfo        = ymgc_toolbox.save_metainfo
    save_message         = ymgc_toolbox.save_message
    show_message         = ymgc_toolbox.show_message
    leave_message        = ymgc_toolbox.leave_message
    add_image            = ymgc_toolbox.add_image
    add_video            = ymgc_toolbox.add_video
    DBHOST      = ymgc_toolbox.DBHOST
    DBUSER      = ymgc_toolbox.DBUSER
    DBPASS      = ymgc_toolbox.DBPASS
    DBCHAR      = ymgc_toolbox.DBCHAR
    TMP_DIR     = ymgc_toolbox.TMP_DIR
    IV_FILE_DIR = ymgc_toolbox.IV_FILE_DIR
    IV_OLD2NEW  = ymgc_toolbox.IV_OLD2NEW
except:
    def gmail(*input1, **input2):
        print 'gmail() cannot be loaded.'
        return
    def cp2www(*input1, **input2):
        print 'cp2www() cannot be loaded.'
        return
    def get_example(*input1, **input2):
        print 'get_example() cannot be loaded.'
        return
    def cPicklecache(*input1, **input2):
        print 'cPicklecache() cannot be loaded.'
        return
    def get_connector_cursor(*input1, **input2):
        print 'get_connector_cursor() cannot be loaded.'
        return
    def insert_to_mysql(*input1, **input2):
        print 'insert_to_mysql() cannot be loaded.'
        return
    def update_mysql(*input1, **input2):
        print 'update_mysql() cannot be loaded.'
        return
    def load_metainfo(*input1, **input2):
        print 'load_metainfo() cannot be loaded.'
        return
    def save_metainfo(*input1, **input2):
        print 'save_metainfo() cannot be loaded.'
        return
    def save_message(*input1, **input2):
        print 'save_message() cannot be loaded.'
        return
    def show_message(*input1, **input2):
        print 'show_message() cannot be loaded.'
        return
    def leave_message(*input1, **input2):
        print 'leave_message() cannot be loaded.'
        return
    DBHOST = None
    DBUSER = None
    DBPASS = None
    DBCHAR = None
    TMP_DIR = '.tmp'
    IV_FILE_DIR = None
    IV_OLD2NEW  = None

class MyTB(ultratb.FormattedTB):
    def __init__(self, mode='Plain', color_scheme='Linux', call_pdb=False,
                 ostream=None,
                 tb_offset=0, long_header=False, include_vars=False,
                 check_cache=None, send_email=False, error_func=None):
        self.send_email   = send_email
        self.color_scheme = color_scheme
        self.error_func   = error_func
        ultratb.FormattedTB.__init__(self, mode=mode,
                                     color_scheme=color_scheme,
                                     call_pdb=call_pdb,
                                     ostream=ostream,
                                     tb_offset=tb_offset,
                                     long_header=long_header,
                                     include_vars=include_vars,
                                     check_cache=check_cache)
    def __call__(self, etype=None, evalue=None, etb=None):
        print get_cur_time()
        if self.send_email and \
           not etype.__name__ == 'KeyboardInterrupt':
            try:
                title = 'Debugger has been called in "%s" on %s.' % (
                    sys.argv[0], os.uname()[1])
                self.set_colors('NoColor')
                body  = self.text(etype, evalue, etb)
                self.set_colors(self.color_scheme)
                gmail(title, body)
            except:
                pass
        # remove a leave_message function from atexit._exithandlers
        try:
            for i in range(len(atexit._exithandlers))[::-1]:
                if 'leave_message' in atexit._exithandlers[i][0].__name__:
                    del atexit._exithandlers[i]
        except:
            pass
        ultratb.FormattedTB.__call__(self, etype=etype,
                                     evalue=evalue, etb=etb)
        if self.error_func is not None:
            try:
                self.error_func()
            except:
                pass

def ip_excepthook(type, value, traceback):
    embed()

def set_debugger_ipython(*arg_list, **arg_dict):
    sys.excepthook = ip_excepthook

def set_debugger(send_email=False, error_func=None):
    if isinstance(sys.excepthook, MyTB):
        return
    sys.excepthook = MyTB(call_pdb=True, send_email=send_email,
                          error_func=error_func)
    print 'MyTB has been set to except hook.', get_cur_time()

def set_debugger_org():
    if not sys.excepthook == sys.__excepthook__:
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def set_debugger_org_frc():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

class MyLogger(object):
    def __init__(self, fobj, stdout=False, stderr=False):
        assert stdout or stderr
        if isinstance(fobj, basestring):
            fobj = open(fobj, 'w')
        self._fo = fobj
        self._stdout = stdout
        self._stderr = stderr

    def __del__(self):
        self._fo.close()

    def write(self, *arg_list, **arg_dict):
        self._fo.write(*arg_list, **arg_dict)
        if self._stdout:
            sys.__stdout__.write(*arg_list, **arg_dict)
        if self._stderr:
            sys.__stderr__.write(*arg_list, **arg_dict)

    def flush(self):
        self._fo.write('\n')
        if self._stdout:
            sys.__stdout__.flush()
        if self._stderr:
            sys.__stderr__.flush()

def output_log(path=None, remove_old=True):
    if path is None:
        cur_fname = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        fname_base = cur_fname.split('/')[-1].split('.')[0]
        log_dir = os.path.expanduser('~/others/logs/python_logs/')
        path = os.path.join(log_dir, 'logs_%s_%s.txt' % (fname_base, get_time_str()))
        makebsdirs_if_missing(path)
        if remove_old:
            file_list = full_listdir(log_dir)
            file_list.sort(key=os.path.getmtime, reverse=False)
            for cur_file in file_list[:-100]:
                os.remove(cur_file)
    fobj = open(path, 'w')
    sys.stdout = MyLogger(fobj, stdout=True)
    sys.stderr = MyLogger(fobj, stderr=True)

class TimeReporter:
    def __init__(self, max_count, interval=1, moving_average=False):
        self.time           = time.time
        self.start_time     = time.time()
        self.max_count      = max_count
        self.cur_count      = 0
        self.prev_time      = time.time()
        self.interval       = interval
        self.moving_average = moving_average
    def report(self, cur_count=None, max_count=None, overwrite=True, prefix=None, postfix=None, interval=None):
        if cur_count is not None:
            self.cur_count = cur_count
        else:
            self.cur_count += 1
        if max_count is None:
            max_count = self.max_count
        cur_time = self.time()
        elapsed  = cur_time - self.start_time
        if self.cur_count <= 0:
            ave_time = float('inf')
        elif self.moving_average and self.cur_count == 1:
            ave_time = float('inf')
            self.ma_prev_time = cur_time
        elif self.moving_average and self.cur_count == 2:
            self.ma_time      = cur_time - self.ma_prev_time
            ave_time          = self.ma_time
            self.ma_prev_time = cur_time
        elif self.moving_average:
            self.ma_time      = self.ma_time * 0.95 + (cur_time - self.ma_prev_time) * 0.05
            ave_time          = self.ma_time
            self.ma_prev_time = cur_time
        else:
            ave_time = elapsed / self.cur_count
        ETA = (max_count - self.cur_count) * ave_time
        print_str = 'count : %d / %d, elapsed time : %f, ETA : %f' % (self.cur_count, self.max_count, elapsed, ETA)
        if prefix is not None:
            print_str = str(prefix) + ' ' + print_str
        if postfix is not None:
            print_str += ' ' + str(postfix)
        this_interval = self.interval
        if interval is not None:
            this_interval = interval
        if cur_time - self.prev_time < this_interval:
            return
        if overwrite and self.cur_count != self.max_count:
            printr(print_str)
            self.prev_time = cur_time
        else:
            print print_str
            self.prev_time = cur_time

def textread(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '').replace('\r', '')
    return lines

def textdump(path, lines, need_asking=False):
    if os.path.exists(path) and need_asking:
        if 'n' == choosebyinput(['Y', 'n'], path + ' exists. Would you replace? [Y/n]'):
            return False
    f = open(path, 'w')
    for i in lines:
        f.write(i + '\n')
    f.close()

def pickleload(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path)
    this_ans = pickle.load(f)
    f.close()
    return this_ans

def pickledump(path, this_dic):
    f = open(path, 'w')
    this_ans = pickle.dump(this_dic, f)
    f.close()

def cPickleload(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path, 'rb')
    this_ans = cPickle.load(f)
    f.close()
    return this_ans

def cPickledump(path, this_dic):
    f = open(path, 'wb')
    this_ans = cPickle.dump(this_dic, f, -1)
    f.close()

def jsonload(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f)
    f.close()

def choosebyinput(cand, message=False):
    if not type(cand) == list and not type(cand) == int:
        print 'The type of cand_list has to be \'list\' or \'int\' .'
        return
    if type(cand) == int:
        cand_list = range(cand)
    if type(cand) == list:
        cand_list = cand
    int_cand_list = []
    for i in cand_list:
        if type(i) == int:
            int_cand_list.append(str(i))
    if message == False:
        message = 'choose by input ['
        for i in int_cand_list:
            message += i + ' / '
        for i in cand_list:
            if not str(i) in int_cand_list:
                message += i + ' / '
        message = message[:-3] + '] : '
    while True:
        your_ans = raw_input(message)
        if your_ans in int_cand_list:
            return int(your_ans)
        if your_ans in cand_list:
            return your_ans

def mv_files(name1, name2, targ_dir=None):
    if targ_dir is None:
        files = os.listdir('.')
    else :
        files = [os.path.join(targ_dir, fname) for fname in os.listdir(targ_dir)]
    for this_file in files:
        if name1 in this_file:
            flg = True
            if os.path.exists(this_file.replace(name1, name2)):
                your_ans = choosebyinput(['Y', 'n'], message=this_file.replace(name1, name2) + ' exists. Would you replace? [Y/n]')
                if your_ans == 'n':
                    flg = False
                    break
                elif your_ans == 'Y':
                    flg = True
                    break
            if flg:
                shutil.move(this_file, this_file.replace(name1, name2))
                print this_file, 'is moved to', this_file.replace(name1, name2)

def find_from_to(str1, start_str, end_str):
    start_num = str1.find(start_str) + len(start_str)
    end_num = str1.find(end_str, start_num)
    return str1[start_num:end_num]

def get_photo(url, fname):
    try:
        urllib.urlretrieve(url, fname)
        urllib.urlcleanup()
        return True
    except IOError:
        return False
    except urllib2.HTTPError:
        return False
    except urllib2.URLError:
        return False
    except httplib.BadStatusLine:
        return False

def get_photos(photos):
    for i in photos:
        threads=[]
        for photo in photos:
            if not 'http' in photo[0]:
                print 'Maybe urls and file names are the opposite. You should switch the indices.'
            t = threading.Thread(target = get_photo,args = (photo[0], photo[1]))
            threads.append(t)
            t.start()

def printr(*targ_str):
    str_to_print = ''
    for temp_str in targ_str:
        str_to_print += str(temp_str) + ' '
    str_to_print = str_to_print[:-1]
    sys.stdout.write(str_to_print + '\r')
    sys.stdout.flush()

def make_red(prt):
    return '\033[91m%s\033[00m' % prt

def emphasise(*targ_str):
    str_to_print = ''
    for temp_str in targ_str:
        str_to_print += str(temp_str) + ' '
    str_to_print = str_to_print[:-1]
    num_repeat = len(str_to_print) / 2 + 1
    print '＿' + '人' * (num_repeat + 1) + '＿'
    print '＞　%s　＜' % make_red(str_to_print)
    print '￣' + 'Y^' * num_repeat + 'Y￣'

emphasize = emphasise # american spelling

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def makedirs_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def makebsdirs_if_missing(f_path):
    makedirs_if_missing(os.path.dirname(f_path) if '/' in f_path else f_path)

def wait_for(targ_path, wait_sec=3):
    wait_flg = False
    temp_count = 0
    if len(targ_path) > 50:
        targ_path_str = '[...]' + targ_path[-45:]
    else:
        targ_path_str = targ_path
    while not os.path.exists(targ_path):
        printr('waiting for appearing %s (Elapsed : %d sec)' % (targ_path_str, temp_count * wait_sec))
        temp_count += 1
        time.sleep(wait_sec)
        wait_flg = True
    if wait_flg:
        time.sleep(wait_sec)
    elif time.time() - os.path.getmtime(targ_path) < wait_sec:
        time.sleep(wait_sec)

def split_inds(all_num, split_num, split_targ):
    assert split_num >= 1
    assert split_targ >= 0
    assert split_targ < split_num
    part = all_num // split_num
    if not split_num == split_targ+1:
        return split_targ * part, (split_targ+1) * part
    else:
        return split_targ * part, all_num

try:
    import numpy as np
    def are_same_vecs(vec_a, vec_b, this_eps1=1e-5, verbose=False):
        if not vec_a.ravel().shape == vec_b.ravel().shape:
            return False
        if np.linalg.norm(vec_a.ravel()) == 0:
            if not np.linalg.norm(vec_b.ravel()) == 0:
                if verbose:
                    print 'assertion failed.'
                    print 'diff norm : %f' % (np.linalg.norm(vec_a.ravel() - vec_b.ravel()))
                return False
        else:
            if not np.linalg.norm(vec_a.ravel() - vec_b.ravel()) / np.linalg.norm(vec_a.ravel()) < this_eps1:
                if verbose:
                    print 'assertion failed.'
                    print 'diff norm : %f' % (np.linalg.norm(vec_a.ravel() - vec_b.ravel()) / np.linalg.norm(vec_a.ravel()))
                return False
        return True
    def comp_vecs(vec_a, vec_b, this_eps1=1e-5):
        assert are_same_vecs(vec_a, vec_b, this_eps1, True)
    def arrayinfo(np_array):
        print 'max: %04f, min: %04f, abs_min: %04f, norm: %04f,' % (np_array.max(), np_array.min(), np.abs(np_array).min(), np.linalg.norm(np_array)),
        print 'dtype: %s,' % np_array.dtype,
        print 'shape: %s,' % str(np_array.shape),
        print

except:
    def comp_vecs(*input1, **input2):
        print 'comp_vecs() cannot be loaded.'
        return
    def arrayinfo(*input1, **input2):
        print 'arrayinfo() cannot be loaded.'
        return

try:
    import Levenshtein
    def search_nn_str(targ_str, str_lists):
        dist = float('inf')
        dist_str = None
        for i in sorted(str_lists):
            cur_dist = Levenshtein.distance(i, targ_str)
            if dist > cur_dist:
                dist = cur_dist
                dist_str = i
        return dist_str
except:
    def search_nn_str(targ_str, str_lists):
        print 'search_nn_str() cannot be imported.'
        return

def flatten(targ_list):
    new_list = copy.deepcopy(targ_list)
    for i in reversed(range(len(new_list))):
        if isinstance(new_list[i], list) or isinstance(new_list[i], tuple):
            new_list[i:i+1] = flatten(new_list[i])
    return new_list

def predict_charset (targ_str):
    targ_charsets = ['utf-8', 'cp932', 'euc-jp', 'iso-2022-jp']
    for targ_charset in targ_charsets:
        try:
            targ_str.decode(targ_charset)
            return targ_charset
        except UnicodeDecodeError:
            pass
    return None

def remove_non_ascii(targ_str, charset=None):
    if charset is not None:
        assert isinstance(targ_str, str)
        targ_str = targ_str.decode(charset)
    else:
        assert isinstance(targ_str, unicode)
    return ''.join([x for x in targ_str if ord(x) < 128]).encode('ascii')

def distribute_values(value_list, host='', port=55555, backlog=10,
                      func_finish=None, serialize=None):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    writefds = set()
    cur_ind = 0
    if serialize is None:
        func_syl = cPickle.dumps
    else:
        func_syl = serialize
    try:
        server_sock.bind((host, port))
        server_sock.listen(backlog)
        while True:
            if len(writefds) > 0:
                wready = select.select([], writefds, [])[1]
            else:
                wready = set()
            for sock in wready:
                if cur_ind < len(value_list):
                    sock.send(func_syl(value_list[cur_ind]))
                    cur_ind += 1
                    print '[distribute_values]: %d / %d' % (
                        cur_ind,
                        len(value_list))
                    if cur_ind == len(value_list):
                        print '[distribute_values]: Finished.'
                elif cur_ind == len(value_list) and func_finish is not None:
                    func_finish()
                    sock.send(func_syl(False))
                    cur_ind += 1
                else:
                    sock.send(func_syl(False))
                sock.close()
                writefds.remove(sock)
            conn, _ = server_sock.accept()
            writefds.add(conn)
    finally:
        for sock in writefds:
            sock.close()
        server_sock.close()

def receive_value(host=None, port=55555, bufsize=4096):
    if host is None:
        host = socket.gethostname()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with closing(sock):
        sock.connect((host, port))
        value = cPickle.loads(sock.recv(bufsize))
    return value

def get_fname_and_lineno(frame_no=2):
    frame = inspect.currentframe(frame_no)
    fname = inspect.getfile(frame)
    cur_loc = frame.f_lineno
    return fname, cur_loc
    # name = '%s (%s)' % (fname, str(cur_loc))

class StopWatch(object):
    def __init__(self):
        self._time = {}
        self._bef_time = {}
        self._last_name = None
    def tic(self, name=None):
        if name is None:
            # frame = inspect.currentframe(1)
            # fname = inspect.getfile(frame)
            # cur_loc = frame.f_lineno
            fname, cur_loc = get_fname_and_lineno()
            name = '%s (%s)' % (fname, str(cur_loc))
        self._bef_time[name] = time.time()
        self._last_name = name
    def toc(self, name=None):
        if name is None:
            name = self._last_name
        self._time[name] = time.time() - self._bef_time[name]
    def show(self, overwrite=False):
        show_str = ''
        for name, elp in self._time.iteritems():
            show_str += '%s: %03.3f, ' % (name, elp)
        if overwrite:
            printr(show_str[:-2])
        else:
            print show_str[:-2]

Timer = StopWatch # deprecated

def get_free_gpu(default_gpu):
    FORMAT = '--format=csv,noheader'
    COM_GPU_UTIL = 'nvidia-smi --query-gpu=index,uuid ' + FORMAT
    COM_GPU_PROCESS = 'nvidia-smi --query-compute-apps=gpu_uuid ' + FORMAT
    uuid2id = {cur_line.split(',')[1].strip(): int(cur_line.split(',')[0])
               for cur_line in commands.getoutput(COM_GPU_UTIL).split('\n')}
    used_gpus = set()
    for cur_line in commands.getoutput(COM_GPU_PROCESS).split('\n'):
        used_gpus.add(cur_line)
    if len(uuid2id) == len(used_gpus):
        return default_gpu
    elif os.uname()[1] == 'konoshiro':
        return 1 - int(uuid2id[list(set(uuid2id.keys()) - used_gpus)[0]])
    else:
        return uuid2id[list(set(uuid2id.keys()) - used_gpus)[0]]

def get_abs_path():
    return os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), __file__)))

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def get_time_str():
    return datetime.datetime.now().strftime('Y%yM%mD%dH%HM%MS%S')

def get_cur_time():
    return str(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

def split_carefully(text, splitter=',', delimiters=['"', "'"]):
    # assertion
    assert isinstance(splitter, str)
    assert not splitter in delimiters
    if not (isinstance(delimiters, list) or isinstance(delimiters, tuple)):
        delimiters = [delimiters]
    for cur_del in delimiters:
        assert len(cur_del) == 1

    cur_ind = 0
    prev_ind = 0
    splitted = []
    is_in_delimiters = False
    cur_del = None
    while cur_ind < len(text):
        if text[cur_ind] in delimiters:
            if text[cur_ind] == cur_del:
                is_in_delimiters = False
                cur_del = None
                cur_ind += 1
                continue
            elif not is_in_delimiters:
                is_in_delimiters = True
                cur_del = text[cur_ind]
                cur_ind += 1
                continue
        if not is_in_delimiters and text[cur_ind] ==  splitter:
            splitted.append(text[prev_ind:cur_ind])
            cur_ind += 1
            prev_ind = cur_ind
            continue
        cur_ind += 1
    splitted.append(text[prev_ind:cur_ind])
    return splitted

def location(depth=0):
    frame = inspect.currentframe(depth+1)
    return frame.f_lineno

class plistMaster(threading.Thread):
    keep_doing = True

    def __init__(self, server_sock, len_list, info_path, verbose):
        self._server_sock = server_sock
        self._len_list = len_list
        self._info_path = info_path
        self._verbose = verbose
        threading.Thread.__init__(self)

    def run(self):
        writefds = set()
        cur_ind = 0
        while self.keep_doing:
            conn, _ = self._server_sock.accept()
            writefds.add(conn)
            if len(writefds) > 0:
                wready = select.select([], writefds, [])[1]
            else:
                wready = set()
            for sock in wready:
                if cur_ind < self._len_list:
                    sock.send(str(cur_ind))
                    cur_ind += 1
                    if self._verbose:
                        print '[distribute_values]: %d / %d' % (cur_ind, self._len_list)
                sock.close()
                writefds.remove(sock)
                if cur_ind == self._len_list:
                    self.keep_doing = False
                    break
        if self._verbose:
            print '[distribute_values]: Finished.'
        for sock in writefds:
            sock.close()
        self._server_sock.close()
        os.remove(self._info_path)

# TODO: show speed when verbose is True
class plist(object):
    _is_master = None
    _master_server = None
    _master_port = None
    _bufsize = 4096
    _backlog = 10
    _mpm = None
    _info_path = None
    _temp_list_ind = None

    def __init__(self, targ_list, verbose=False, show_pace=False):
        self._verbose = verbose
        self._targ_list = targ_list
        self._get_master_information()
        self._show_pace = show_pace
        if self._show_pace:
            self.TR = TimeReporter(len(targ_list))
        if self.is_master:
            # work as a master node
            self._start_thread()

    def __len__(self):
        return len(self._targ_list)

    @property
    def is_master(self):
        if self._is_master is not None:
            return self._is_master
        self._is_master = False
        if not self._can_find_master(): # if cannot find master node
            self._is_master = True
        return self._is_master

    def _can_find_master(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self._master_server, self._master_port))
            self._temp_list_ind = int(sock.recv(self._bufsize))
            sock.close()
        except socket.error as err:
            if err.errno == 111:
                return False
            print err
            raise Exception('Unknown Error. Inspect err.')
        return True

    def _start_thread(self):
        # find a port number which is not being used
        for i in range(3):
            cur_port = random.randint(49200, 65500)
            try:
                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_sock.bind(('', cur_port))
                server_sock.listen(self._backlog)
                break
            except socket.error as err:
                if err.errno == 98:
                    continue
                print err
                raise Exception('Unknown Error. Inspect err.')
            if i == 2:
                raise Exception('Has reached maximum trial number.')
        self._master_server = socket.gethostname()
        self._master_port = cur_port

        # pack information and save as a file on self._info_path
        # - "tai 55555"
        textdump(self._info_path, ['%s %d' % (self._master_server, self._master_port)])

        # start thread
        self._mpm = plistMaster(server_sock, len(self._targ_list), self._info_path, self._verbose)
        self._mpm.start()

        # register function as excepthook for keyboardinterrupt
        cur_excepthook = sys.excepthook
        def kill_thread(*input1, **input2):
            cur_excepthook(*input1, **input2)
            self._mpm.keep_doing = False
            try:
                self.next()
            except StopIteration:
                pass
        sys.excepthook = kill_thread

    def __iter__(self):
        return self

    def next(self):
        if self._temp_list_ind is not None:
            list_ind = self._temp_list_ind
            self._temp_list_ind = None
            if self._show_pace:
                self.TR.report(list_ind)
            return self._targ_list[list_ind]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self._master_server, self._master_port))
            list_ind = int(sock.recv(self._bufsize))
        except socket.error as err:
            if err.errno == 111:
                raise StopIteration()
            print err
            raise Exception('Unknown Error. Inspect err.')
        sock.close()
        if self._show_pace:
            self.TR.report(list_ind)
        return self._targ_list[list_ind]

    def _get_master_information(self):
        cur_filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        cur_dirname = os.path.abspath(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
        self._info_path = os.path.join(cur_dirname, '.myparallel', '%s_%d.txt' % (cur_filename, location(2)))
        if not os.path.exists(self._info_path):
            makebsdirs_if_missing(self._info_path)
            self._is_master = True
            return
        file_content = textread(self._info_path)
        assert len(file_content) == 1
        self._master_server, self._master_port = file_content[0].strip().split()
        self._master_port = int(self._master_port)

class prange(plist):
    _matlab_mode = False
    def __init__(self, *input1, **input2):
        super(prange, self).__init__(range(*input1, **input2))

    def matlab_mode(self):
        self._matlab_mode = True

    def next(self):
        if self._matlab_mode:
            try:
                return super(prange, self).next()
            except StopIteration:
                return -1
        else:
            return super(prange, self).next()

def full_listdir(dir_name):
    return [os.path.join(dir_name, i) for i in os.listdir(dir_name)]

class tictoc(object):
    def __init__(self, targ_list):
        self._targ_list = targ_list
        self._list_ind = -1
        self._TR = TimeReporter(len(targ_list))
    def __iter__(self):
        return self
    def next(self):
        self._list_ind += 1
        if self._list_ind > 0:
            self._TR.report()
        if self._list_ind == len(self._targ_list):
            print
            raise StopIteration()
        return self._targ_list[self._list_ind]

def is_updated_within(filename, days=0, hours=0):
    stat = os.stat(filename)
    last_modified = stat.st_mtime
    dt_now = datetime.datetime.now()
    dt_file = datetime.datetime.fromtimestamp(last_modified)
    timediff = dt_now - dt_file
    return timediff < datetime.timedelta(days=days, hours=hours)

def execute_command(cmd, lim=None, shell=False):
    if shell:
        popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        popen = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if lim is not None:
        assert isinstance(lim, int)
        for i in range(lim):
            if popen.poll() != 0:
                time.sleep(1)
        if not popen.poll() in [0, 127]:
            try:
                popen.terminate()
            except:
                pass
        if popen.poll() is None:
            return '', ''
    res, err = popen.communicate()
    return res[:-1], err[:-1]

ONCE_PRINTED = set()

def print_once(*targ_str):
    frame = inspect.currentframe(1)
    fname = inspect.getfile(frame)
    cur_loc = frame.f_lineno
    cur_key = fname + str(cur_loc)
    if cur_key in ONCE_PRINTED:
        return
    else:
        ONCE_PRINTED.add(cur_key)
    str_to_print = ''
    for temp_str in targ_str:
        str_to_print += str(temp_str) + ' '
    str_to_print = str_to_print[:-1]
    print str_to_print

try:
    import h5py
    MAT_H5PY_FLG = defaultdict(bool)
    class MatFile(object):
        h5py_flg = False
        f = None
        try_count = 0
        def __init__(self, fname, cur_key):
            self.h5py_flg = MAT_H5PY_FLG[cur_key]
            if self.h5py_flg:
                self.open_with_h5py(fname, cur_key)
            if not self.h5py_flg:
                self.open_with_sio(fname, cur_key)

        def open_with_h5py(self, fname, cur_key):
            assert self.try_count < 2
            try:
                self.try_count += 1
                self.f = h5py.File(fname, 'r')
                return True
            except:
                self.h5py_flg = not self.h5py_flg
                MAT_H5PY_FLG[cur_key] = self.h5py_flg
                assert self.open_with_sio(fname, cur_key)
                return True

        def open_with_sio(self, fname, cur_key):
            assert self.try_count < 2
            try:
                self.try_count += 1
                self.f = sio.loadmat(fname)
                return True
            except:
                self.h5py_flg = not self.h5py_flg
                MAT_H5PY_FLG[cur_key] = self.h5py_flg
                assert self.open_with_h5py(fname, cur_key)
                return True

        def access(self, varname):
            if self.h5py_flg:
                return np.array(self.f[varname])
            else:
                return self.f[varname]

        def close(self):
            if self.h5py_flg:
                self.f.close()

    def load_mat_file(path, varname):
        frame = inspect.currentframe(1)
        fname = inspect.getfile(frame)
        cur_loc = frame.f_lineno
        cur_key = fname + str(cur_loc)

        mf = MatFile(path, cur_key)
        if not isinstance(varname, (list, tuple)):
            data = mf.access(varname)
            mf.close()
            return data
        data = {}
        for cur_var in varname:
            data[cur_var] = mf.access(cur_var)
        mf.close()
        return data
except:
    pass

def canbeinstance(targ_value, targ_type):
    try:
        targ_type(targ_value)
        return True
    except:
        return False

def show_parser(parser):
    args = parser.parse_args()
    for arg_key in dir(args):
        if arg_key.startswith('_'):
            continue
        cur_val = getattr(args, arg_key)
        org_val = parser.get_default(arg_key)
        if cur_val == org_val:
            continue
        print('# {}: {} -> {}'.format(arg_key, org_val, cur_val))

def show_info(var):
    from chainer import Variable
    if isinstance(var, Variable):
        targ_list = ['shape', 'dtype']
    if isinstance(var, (np.ndarray, cp.array)):
        targ_list = ['shape', 'dtype']
    for i in targ_list:
        print '[%s] %s' % (i, str(getattr(var, i)))
    if isinstance(var, Variable):
        print '[%s] %s' % (i, str(type(var.data)))

def analize_mat(A, B, info=''):
    A = np.asarray(A)
    B = np.asarray(B)
    abs_mat = np.abs(A - B)
    abs_max = np.max(abs_mat)
    abs_mean = np.mean(abs_mat)
    abs_std = np.std(abs_mat)
    r, p = pearsonr(A.ravel(), B.ravel())
    same = are_same_vecs(A, B)
    if not info.endswith(' '):
        info = '[%s] ' % info
    print '%smax(|A-B|): %05f, mean(|A-B|): %05f, std(|A-B|): %05f, corr: %05f, same?: %s' % (info, abs_max, abs_mean, abs_std, r, str(same))
    return

def assert_abs(a, b=1e-5):
    assert abs(a) < b

def get_file_body(fname):
    return '.'.join(fname.split('/')[-1].split('.')[:-1])
