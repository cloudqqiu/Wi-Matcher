import haversine as hs
from pyhanlp import *
import re
import Levenshtein as lv
import math


def jaccard(s, t):
    return len(set(s) & set(t)) / len(set(s) | set(t))


def edit_dis(s, t):
    return lv.distance(s, t)


def cos_onehot_vector(s, t):
    return len(set(s) & set(t)) / (math.sqrt(len(s)) * math.sqrt((len(t))))


def get_ngram(s, n=3, need_short_slice=True):
    assert n > 0 and type(n) == int
    if len(s) < n:
        if need_short_slice:
            return [s]
        else:
            return []
    else:
        return [s[i:i+n] for i in range(len(s) - n + 1)]


def exchange_lag_lng(s):
    first, second = s.split(',')
    return '{},{}'.format(second, first)


def distance_poi(a, b):  # tuple (lat, lon)
    return hs.haversine(a, b, unit='m')


def chinese2shengmu(text, short=False):
    Pinyin = JClass("com.hankcs.hanlp.dictionary.py.Pinyin")
    pinyin_list = HanLP.convertToPinyinList(text)
    s = str()
    for index, pinyin in enumerate(pinyin_list):
        if pinyin.getShengmu().toString() != 'none':
            if short:
                s += pinyin.getShengmu().toString()[0]
            else:
                s += pinyin.getShengmu().toString()
        else:
            s += text[index]
    return s


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


def chinese2pyandsm(text, short=False):
    Pinyin = JClass("com.hankcs.hanlp.dictionary.py.Pinyin")

    # from jpype import java
    # String = java.lang.String
    # java_string = String(text.encode(), 'UTF8')
    # string_from_java = str(java_string).encode('utf-16', errors='surrogatepass').decode('utf-16')

    pinyin_list = HanLP.convertToPinyinList(text)  # HanLP.convertToTraditionalChinese
    py, sm = str(), str()
    for index, pinyin in enumerate(pinyin_list):
        if pinyin.getPinyinWithoutTone() != 'none':
            py += pinyin.getPinyinWithoutTone()
            if pinyin.getShengmu().toString() != 'none':
                if short:
                    sm += pinyin.getShengmu().toString()[0]
                else:
                    sm += pinyin.getShengmu().toString()
            else:
                sm += pinyin.getPinyinWithoutTone()[0]
        else:
            py += text[index]
            sm += text[index]

    return py, sm


def filter_emoji(desstr, restr=''):
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)


def clean_str(pattern, perserve, s):
    start = 0
    result = str()
    while start < len(s):
        cut = re.search(pattern, s[start:])
        if cut:
            cut_index = cut.span()
            result += s[start:start + cut_index[0]]
            for i in range(cut_index[0], cut_index[1]):
                if s[start + i] in perserve:
                    result += s[start + i]
            start += cut_index[1]
        else:
            result += s[start:]
            start = len(s)
    return result


def get_stopwords(stop_file, perserve):
    stopwords = list()
    with open(stop_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # s = set(line.strip())  # 停用词不能与保留字符有重合字符
            # if len(s & perserve) == 0:
            #     stopwords.append(line.strip())
            # # else:
            # #     print(line.strip())
            s = line.strip()  # 只要停用词不与保留字符完全相同
            if s not in perserve:
                stopwords.append(s)
    return stopwords


def drop_stopwords(s, stopwords):
    for w in stopwords:
        s = s.replace(w, '')
    return s


def get_rid_of_zero(s):
    # s: string 12.300
    flag = False
    for i in range(len(s) - 1, -1, -1):
        if s[i] == '.':
            if flag:
                return s[:i]
            else:
                return s
        elif s[i] == '0':
            flag = True
        elif flag:
            return s[:i+1]
        else:
            return s



