import math

import numpy as np
import pandas as pd
from pyhanlp import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import models.model_config as config
from models.common import utils


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


def process_zh_data(pois, save=False):
    # import api_dp_analyse as adp
    # import api_amap_analyse as aamap
    # amap_shop = aamap.read_new_shop_list()
    # dp_shop = adp.read_shop_list()
    shop_path = '../src/around'
    suffix = ['cos', 'cover', 'tfidf', 'local_tfidf', 'cos_tfidf', 'cos_local_tfidf']
    col = ['ssid', 'name', 'label', 'ssid_len', 'ssid_token_len', 'names_len_mean', 'names_tokens_count_mean'] + \
          ['name_char_trigram_' + i for i in ['cos', 'cover', 'cos_tfidf']] + \
          ['name_char_' + i for i in ['cos', 'cover']] + \
          ['name_token_' + i for i in ['cos', 'local_tfidf', 'cos_local_tfidf']]
    total_df = pd.DataFrame(columns=col)
    total_text_py = list()
    for index, poi in enumerate(pois):
        # if poi != '39.88892,116.32670':  # test
        #     continue
        try:
            pos = pd.read_csv(f'{config.zh_yandexmatcher_data_path}/prepro/pos_{poi}.csv')
            neg = pd.read_csv(f'{config.zh_yandexmatcher_data_path}/prepro/neg_{poi}.csv')
        except Exception as e:
            print(e)
        this_samples = pd.concat([pos, neg])
        print('Processing : {}; Pos Neg Read'.format(poi))
        shop_list = dict()
        exist = 0
        with open('{}/new amap/{}'.format(shop_path, poi), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                shop = eval(line.strip())
                if shop_list.__contains__(shop['id']):
                    exist += 1
                    # print('shop = {} existed!'.format(content[2]))
                else:
                    shop_list[shop['id']] = shop
        with open('{}/new dp/data/{}.txt'.format(shop_path, '-'.join([get_rid_of_zero(_) for _ in poi.split(',')])), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                shop = line.strip().split('\t')
                if shop_list.__contains__(shop[2]):
                    exist += 1
                    # print('shop = {} existed!'.format(content[2]))
                else:
                    shop_list[shop[2]] = shop

        text_list_py = list()
        match_shop_id_list = list()  # 带|的取高德
        match_shop_text_list = list()
        for i, row in this_samples.iterrows():
            text_list_py.append(chinese2pinyin(row['wifi']))
            if '|' in row['match']:
                temp_id = row['match'].split('|')[0]
            else:
                temp_id = row['match']
            match_shop_id_list.append(temp_id)
            if shop_list.__contains__(temp_id):
                if temp_id[0] == 'B':
                    temp_text = shop_list[temp_id]['name']
                else:
                    temp_text = shop_list[temp_id][3]
            else:
                print('not found shop_id {}. set Nan'.format(temp_id))
                temp_text = 'Nan'
            match_shop_text_list.append(temp_text)
            temp_text_py = chinese2pinyin(temp_text)
            if temp_text_py not in text_list_py:
                text_list_py.append(temp_text_py)
        print(text_list_py)
        total_text_py.extend(text_list_py)

        token_X = CountVectorizer().fit_transform(text_list_py)
        token_tfidf = TfidfTransformer().fit_transform(token_X)
        token_array = token_X.toarray()
        token_tfidf_cossim = cosine_similarity(token_tfidf)

        temp_samples = pd.DataFrame(columns=tuple(col))
        for index, row in this_samples.iterrows():
            temp_sample = dict()
            temp_sample['ssid'] = chinese2pinyin(row['wifi'])
            temp_sample['name'] = chinese2pinyin(match_shop_text_list[index])
            temp_sample['label'] = row['label']
            temp_sample['ssid_len'] = len(row['wifi'])
            temp_sample['ssid_token_len'] = len(row['wifi'].split(' '))
            temp_sample['names_len_mean'] = len(temp_sample['name'])
            temp_sample['names_tokens_count_mean'] = len(temp_sample['name'].split(' '))
            temp_sample['name_char_trigram_cos'] = cos_onehot_vector(
                get_ngram(temp_sample['ssid'], 3), get_ngram(temp_sample['name'], 3))
            temp_sample['name_char_trigram_cover'] = \
                len(set(get_ngram(temp_sample['ssid'], 3)) & set(get_ngram(temp_sample['name'], 3))) \
                / len(set(get_ngram(temp_sample['name'], 3)))
            temp_sample['name_char_cos'] = cos_onehot_vector(set(temp_sample['ssid']), set(temp_sample['name']))
            temp_sample['name_char_cover'] = len(set(temp_sample['ssid']) & set(temp_sample['name'])) \
                / len(set(temp_sample['name']))
            temp_sample['name_token_cos'] = cos_onehot_vector(
                set(temp_sample['ssid'].split(' ')), set(temp_sample['name'].split(' ')))
            temp_sample['name_token_local_tfidf'] = sum(
                a * b for a, b in zip(token_array[text_list_py.index(temp_sample['ssid'])],
                                      token_array[text_list_py.index(temp_sample['name'])]))
            temp_sample['name_token_cos_local_tfidf'] = token_tfidf_cossim[text_list_py.index(temp_sample['ssid'])][text_list_py.index(temp_sample['name'])]
            temp_samples = temp_samples.append(pd.DataFrame(temp_sample, index=[0]), ignore_index=True)
        total_df = pd.concat([total_df, temp_samples], ignore_index=True)
    trigram_X = CountVectorizer(analyzer='char', ngram_range=(3, 3)).fit_transform(total_text_py)
    trigram_tfidf = TfidfTransformer().fit_transform(trigram_X)
    trigram_tfidf_cossim = cosine_similarity(trigram_tfidf)
    for i in total_df.index.values.tolist():
        ssid, name = total_df.loc[i, ['ssid', 'name']]
        total_df.loc[i, 'name_char_trigram_cos_tfidf'] = trigram_tfidf_cossim[total_text_py.index(ssid)][total_text_py.index(name)]
    if save:
        total_df.to_csv(f'{config.zh_yandexmatcher_data_path}/dataset.csv')
    return total_df


# To avoid numpy.core._exceptions.MemoryError
def cosine_similarity_n_space(m1, m2, batch_size=100):
    assert m1.shape[1] == m2.shape[1]
    ret = np.ndarray((m1.shape[0], m2.shape[0]), dtype='float16')
    for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break # cause I'm too lazy to elegantly handle edge cases
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) # rows is O(1) size
        ret[start: end] = sim
    return ret


def process_ru_data(use_orig=False, save=True, use_lbs=False):
    data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
    data['name'] = data.apply(utils.select_first_name, axis=1)
    data.rename(columns={'target': 'label'}, inplace=True)

    if not use_orig:
        # ssid_len and ssid_token_len use original data
        # name_tokens_local_tfidf and name_tokens_cosine_local_tfidf use original data
        all_ssid_venue_text = data['name'].tolist() + data['ssid'].tolist()
        trigram_X = CountVectorizer(analyzer='char', ngram_range=(3, 3)).fit_transform(all_ssid_venue_text)
        trigram_tfidf = TfidfTransformer().fit_transform(trigram_X)
        # trigram_tfidf_cossim = cosine_similarity(trigram_tfidf, dense_output=False)
        data['name_char_trigrams_cosine_tfidf'] = data.apply(lambda row: cosine_similarity(trigram_tfidf[all_ssid_venue_text.index(row['ssid'])], trigram_tfidf[all_ssid_venue_text.index(row['name'])]), axis=1)

        data['name_len'] = data.apply(lambda row: len(row['name']), axis=1)
        data['name_tokens_count_mean'] = data.apply(lambda row: len(row['name'].split(' ')), axis=1)
        data['name_char_trigrams_cosine'] = data.apply(lambda row: cos_onehot_vector(get_ngram(row['ssid'], 3), get_ngram(row['name'], 3)), axis=1)
        data['name_char_trigrams_coverage'] = data.apply(lambda row: len(set(get_ngram(row['ssid'], 3)) & set(get_ngram(row['name'], 3))) / len(set(get_ngram(row['name'], 3))), axis=1)
        data['name_chars_cosine'] = data.apply(lambda row: cos_onehot_vector(set(row['ssid']), set(row['name'])), axis=1)
        data['name_chars_coverage'] = data.apply(lambda row: len(set(row['ssid']) & set(row['name'])) / len(set(row['name'])), axis=1)
        data['name_tokens_cosine'] = data.apply(lambda row: cos_onehot_vector(set(row['ssid'].split(' ')), set(row['name'].split(' '))), axis=1)
        if use_lbs:
            data = data[['ssid', 'name', 'label',
                         'ssid_len', 'ssid_tokenlen', 'names_tokens_local_tfidf', 'names_tokens_cosine_local_tfidf',
                         'name_len', 'name_tokens_count_mean', 'name_char_trigrams_cosine', 'name_char_trigrams_coverage',
                         'name_chars_cosine', 'name_chars_coverage', 'name_tokens_cosine', 'name_char_trigrams_cosine_tfidf']]
        else:
            data = data[['ssid', 'name', 'label',
                         'ssid_len', 'ssid_tokenlen',
                         'name_len', 'name_tokens_count_mean', 'name_char_trigrams_cosine',
                         'name_char_trigrams_coverage',
                         'name_chars_cosine', 'name_chars_coverage', 'name_tokens_cosine',
                         'name_char_trigrams_cosine_tfidf']]
    else:
        if use_lbs:
            data = data[['ssid', 'name', 'label',
                         'ssid_len', 'ssid_tokenlen', 'names_tokens_local_tfidf', 'names_tokens_cosine_local_tfidf',
                         'names_len_mean', 'names_tokens_count_mean', 'names_char_trigrams_cosine', 'names_char_trigrams_coverage',
                         'names_chars_cosine', 'names_chars_coverage', 'names_tokens_cosine', 'names_char_trigrams_cosine_tfidf']]
        else:
            data = data[['ssid', 'name', 'label',
                         'ssid_len', 'ssid_tokenlen',
                         'names_len_mean', 'names_tokens_count_mean', 'names_char_trigrams_cosine',
                         'names_char_trigrams_coverage',
                         'names_chars_cosine', 'names_chars_coverage', 'names_tokens_cosine',
                         'names_char_trigrams_cosine_tfidf']]
    if save:
        if use_lbs:
            data.to_csv(f'{config.ru_yandexmatcher_data_path}/dataset_lbs.csv', index=1)
        else:
            data.to_csv(f'{config.ru_yandexmatcher_data_path}/dataset_no_lbs.csv', index=1)

if __name__=='__main__':
    pois = ['39.92451,116.51533', '39.93483,116.45241',  # 这两个是第一批
            '39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
            '39.96333,116.45187', '39.98850,116.41674', '40.00034,116.46960']
    # process_zh_data(pois, save=True)
    process_ru_data(use_lbs=False)