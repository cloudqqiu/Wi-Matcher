import argparse
import math
import time
import os
import sys
sys.path.append(os.getcwd()+"/../..")

from collections import Counter
from timeit import default_timer as timer
import models.common.utils as utils


def VSM(data, mod='char', df=False, thres=0.5):
    assert mod == 'char' or mod == 'gram'
    print('VSM {}-model tf{} thres={} dataset={}'.format(mod, '-idf' if df else '', thres, data))

    timer_log = utils.Timer()

    timer_log.start()

    start_time_data_preprocess = timer()
    raw_data = utils.load_data(data)
    end_time_data_preprocess = timer()
    print(f"# Timer - Data Preprocess - {end_time_data_preprocess - start_time_data_preprocess}")

    timer_log.log('Dataset-load')

    start_time_model_train = timer()
    c_wifi, c_shop = list(), list()
    c_frequent = dict()
    if df:
        df_set = set()
    for index, i in raw_data.iterrows():
        wifi, shop = i['ssid'], i['venue']
        # wifi, shop = i['ltable_pinyin'], i['rtable_pinyin']
        if df:
            df_set.add(wifi)
            df_set.add(shop)
        if mod == 'char':
            w_d, s_d = Counter(wifi), Counter(shop)
            for c in w_d.keys():
                w_d[c] /= len(wifi)
                if c_frequent.__contains__(c):
                    if wifi not in c_frequent[c]:
                        c_frequent[c].add(wifi)
                else:
                    c_frequent[c] = {wifi}
            for c in s_d.keys():
                s_d[c] /= len(shop)
                if c_frequent.__contains__(c):
                    if wifi not in c_frequent[c]:
                        c_frequent[c].add(shop)
                else:
                    c_frequent[c] = {shop}
        elif mod == 'gram':
            w, s = utils.get_ngram(wifi, 3, True), utils.get_ngram(shop, 3, True)
            w_d, s_d = Counter(w), Counter(s)
            for c in w_d.keys():
                w_d[c] /= len(w)
                if c_frequent.__contains__(c):
                    if wifi not in c_frequent[c]:
                        c_frequent[c].add(wifi)
                else:
                    c_frequent[c] = {wifi}
            for c in s_d.keys():
                s_d[c] /= len(s)
                if c_frequent.__contains__(c):
                    if wifi not in c_frequent[c]:
                        c_frequent[c].add(shop)
                else:
                    c_frequent[c] = {shop}
        c_wifi.append(w_d)
        c_shop.append(s_d)
    end_time_model_train = timer()
    print(f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

    start_time_model_predict = timer()
    tp, fp, fn = 0, 0, 0
    for index, i in raw_data.iterrows():
        temp_sum, w_sum, s_sum = 0, 0, 0
        for j in c_wifi[index].keys():
            if df:
                c_wifi[index][j] *= math.log(len(df_set) / len(c_frequent[j]), 10)
            w_sum += math.pow(c_wifi[index][j], 2)
        for j in c_shop[index].keys():
            if df:
                c_shop[index][j] *= math.log(len(df_set) / len(c_frequent[j]), 10)
            s_sum += math.pow(c_shop[index][j], 2)
            if c_wifi[index].__contains__(j):
                temp_sum += c_wifi[index][j] * c_shop[index][j]
        if w_sum == 0 or s_sum == 0:
            print('aaa')
        temp_sum /= math.sqrt(w_sum) * math.sqrt(s_sum)
        if temp_sum > thres:
            if i['label'] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if i['label'] == 1:
                fn += 1

    timer_log.stop('Model-prediction')
    end_time_model_predict = timer()
    print(
        f"# Timer - Model Predict - Samples: {len(raw_data)} - {end_time_model_predict - start_time_model_predict}")

    print('True-Positive:', tp, '\nFalse-Positive:', fp, '\nFalse-Negative:',fn)
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print('micro P:', precision)
        print('micro R:', recall)
        print('micro F1:', f1)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VSM model")
    parser.add_argument('--dataset', '-d', required=True, choices=['ru', 'zh'], help='Dataset zh or ru')
    parser.add_argument('--mode', '-m', default='gram', choices=['char', 'gram'], help='char level or 3-gram level')
    parser.add_argument('--threshold', '-t', default=0.5
                        , type=float, help='char level or 3-gram level')
    parser.add_argument('--df', '-f', default=True)

    args = parser.parse_args()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    VSM(data=args.dataset, mod=args.mode, df=args.df, thres=args.threshold)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))