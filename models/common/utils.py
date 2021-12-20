import os
import sys
import time
from keras.layers import *
import tensorflow as tf
import pandas as pd
import Levenshtein as lv
from pyhanlp import *
from keras import backend as K

import models.model_config as config


def select_first_name(row):
    # SSID2ORG dataset 问题： 两个引号、一个字符串中可能有逗号、可能有\u字符（暂时没发现影响）
    return row['names'][2:-2].split('","')[0].replace('\\', '')

def load_data(src):
    assert src in ['zh', 'ru'], "Data loader are limited to 'zh' and 'ru'"

    if src == 'ru':
        print('USE Russian DATASET')
        data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
        data['venue'] = data.apply(select_first_name, axis=1)
        data = data[['ssid', 'venue', 'target']]
        data.rename(columns={'target':'label'}, inplace=True)
    elif src == 'zh':
        print('USE Chinese DATASET')
        data = pd.read_csv(config.zh_base_dataset, delimiter=',', header=0, low_memory=False, encoding='utf-8')
        data = data[['ltable_pinyin', 'rtable_pinyin', 'label']]
        data.rename(columns={'ltable_pinyin':'ssid', 'rtable_pinyin': 'venue'}, inplace=True)
    return data


def get_ngram(s, n=3, need_short_slice=True):
    assert n > 0 and type(n) == int
    if len(s) < n:
        if need_short_slice:
            return [s]
        else:
            return []
    else:
        return [s[i:i + n] for i in range(len(s) - n + 1)]


class Timer():
    def __init__(self):
        self._time_start = 0
        self._time_stop = 0
        self._time_log = []
        self.print_format = 3

    def start(self):
        print(f'### Timer - Timer started.')
        self._time_start = time.time()
        self._time_log.append(self._time_start)

    def log(self, info='last'):
        log_time = time.time()
        print(f'### Timer - {info} part costed {round(log_time - self._time_log[-1], self.print_format)} s |'
              f' All {round(log_time - self._time_start, self.print_format)} s')
        self._time_log.append(log_time)

    def stop(self, info='last'):
        self._time_stop = time.time()
        self._time_log.append(self._time_stop)
        print(f'### Timer - {info} part costed {round(self._time_log[-1] - self._time_log[-2], self.print_format)} s |'
              f' All {round(self._time_stop - self._time_start, self.print_format)} s')
        print(f'###       - timestamps {[round(x, self.print_format) for x in self._time_log]}')


def get_softmax(tensor):
    return K.softmax(tensor)


def get_softmax_row(tensor):
    return K.softmax(tensor, axis=1)


def get_mean(tensor):
    return K.mean(tensor)


def get_abs(tensor):
    return K.abs(tensor)


def get_l2(tensor):
    return K.l2_normalize(tensor, axis=-1)


def get_std(tensor, axis):
    return K.std(tensor, axis=axis)


def get_repeat(tensor, nums):
    return K.repeat(tensor, n=nums)


def get_max(tensor, axis):
    return K.max(tensor, axis=axis)


def get_transpose(tensor):
    return tf.transpose(tensor, [0, 2, 1])


def softmax10avg(tensors):
    # print(len(tensors))
    # ts = K.concatenate(tensors[1:], axis=-1)
    # print(K.int_shape(ts))
    # ts = K.reshape(ts, (MAX_SR_NUM, NN_DIM))
    ts = K.stack(tensors[1:], axis=1)
    # print(K.int_shape(ts))
    # ts_transpose = K.reshape(ts, (NN_DIM, MAX_SR_NUM))
    ts_transpose = K.permute_dimensions(ts, (0, 2, 1))
    # print(K.int_shape(ts_transpose))
    weights = K.dot(tensors[0], ts_transpose)
    # print(K.int_shape(weights))
    weights = K.squeeze(weights, axis=1)
    # print(K.int_shape(weights))
    sfm_weights = K.softmax(weights)
    print(K.int_shape(sfm_weights))
    # r = K.dot(sfm_weights, ts)
    # print(K.int_shape(r))
    # r = K.squeeze(r, axis=1)
    # print(K.int_shape(r))
    return sfm_weights


def get_stack(tensors):
    if K.is_tensor(tensors):
        return K.expand_dims(tensors, 1)
    return K.stack(tensors, axis=1)

def get_stack0(tensors):
    return K.stack(tensors, axis=0)


def squeeze(tensor):
    return K.squeeze(tensor, axis=1)


def jaccard(s, t):
    return len(set(s) & set(t)) / len(set(s) | set(t))


def edit_dis(s, t):
    return lv.distance(s, t)


def chinese2pinyin(text):
    Pinyin = JClass("com.hankcs.hanlp.dictionary.py.Pinyin")
    pinyin_list = HanLP.convertToPinyinList(text)
    s = str()
    for index, pinyin in enumerate(pinyin_list):
        if pinyin.getShengmu().toString() != 'none':
            s += pinyin.getPinyinWithoutTone()
        else:
            s += text[index]
    return s

if __name__=='__main__':
    # use = pd.read_csv(f'{config.zh_data_path}/match_use.csv')
    # use = use[['ltable_pinyin', 'rtable_pinyin', 'label']]
    # paper = pd.read_csv(f'{config.zh_data_path}/match_paper.csv')
    # paper = paper[['ltable_pinyin', 'rtable_pinyin', 'label']]
    t = Timer()
    t.start()
    load_data('zh')
    t.log('ZH-dataset-load')
    load_data('ru')
    t.stop('RU-dataset-loader')
