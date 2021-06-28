import pro_func as pf
import Levenshtein as lv
from collections import Counter
ngram = 3
ex_path = '../src/experiment'


def our_neg_stratagy(a, b):
    pinyin_a, pinyin_b = pf.chinese2pinyin(a), pf.chinese2pinyin(b)
    pinyin_a_gram, pinyin_b_gram = [pinyin_a[i:i + ngram] for i in range(len(pinyin_a) - ngram + 1)], \
                                   [pinyin_b[i:i + ngram] for i in range(len(pinyin_b) - ngram + 1)]
    overlap = pf.jaccard(pinyin_a_gram, pinyin_b_gram)  # / math.log(len(row['pinyin']))
    overlap2 = pf.jaccard(set(a), set(b))
    score = (overlap + overlap2) * 100.0 + lv.distance(pinyin_a, pinyin_b)
    return score


def process_finder_result(filename):
    wifi_dict, wifi_pos_score = dict(), dict()
    with open('{}/colmatch/result/{}.log'.format(ex_path, filename), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            label, wifi, shop, score = line.strip().split('\t')
            temp_sim_score = our_neg_stratagy(wifi, shop)
            if label[0] == '1':
                if wifi_dict.__contains__(wifi):
                    wifi_dict[wifi].append((label, shop, score, temp_sim_score))
                else:
                    wifi_dict[wifi] = [(label, shop, score, temp_sim_score)]
                if wifi_pos_score.__contains__(wifi):
                    wifi_pos_score[wifi] = min(wifi_pos_score[wifi], temp_sim_score)
                else:
                    wifi_pos_score[wifi] = temp_sim_score
            else:
                if temp_sim_score >= wifi_pos_score[wifi]:
                    wifi_dict[wifi].append((label, shop, score, temp_sim_score))
    with open('{}/colmatch/process_result/pcs_{}.log'.format(ex_path, filename), 'w', encoding='utf-8') as f:
        for wifi in wifi_dict.keys():
            for i in wifi_dict[wifi]:
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(i[0], wifi, i[1], i[2], i[3]))
            counter = Counter([j[0] for j in wifi_dict[wifi]])
            f.write('{}\t{}\n'.format(wifi, counter))


def process_finder_result_topk(filename, process_k=50):
    wifi_dict, wifi_pos_score = dict(), dict()
    with open('{}/colmatch/result/{}.log'.format(ex_path, filename), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        temp_wifi = ''
        temp_count = 0
        for line in lines:
            label, wifi, shop, score = line.strip().split('\t')
            if not temp_wifi:
                temp_wifi = wifi
            temp_sim_score = our_neg_stratagy(wifi, shop)
            if label[0] == '1':
                if temp_wifi != wifi:
                    temp_wifi = wifi
                    temp_count = 0
                if wifi_dict.__contains__(wifi):
                    wifi_dict[wifi].append((label, shop, score, temp_sim_score))
                else:
                    wifi_dict[wifi] = [(label, shop, score, temp_sim_score)]
            else:
                if wifi == temp_wifi:
                    temp_count += 1
                    if temp_count <= process_k:
                        wifi_dict[wifi].append((label, shop, score, temp_sim_score))

    with open('{}/colmatch/process_result/pcs_k{}_{}.log'.format(ex_path, process_k, filename), 'w', encoding='utf-8') as f:
        for wifi in wifi_dict.keys():
            for i in wifi_dict[wifi]:
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(i[0], wifi, i[1], i[2], i[3]))
            counter = Counter([j[0] for j in wifi_dict[wifi]])
            f.write('{}\t{}\n'.format(wifi, counter))


if __name__ == '__main__':
    fname = '39.88892,116.32670_10_simple_bi_gruSun May 24 08 38 25 2020'

    # process_finder_result(fname)
    process_finder_result_topk(fname, process_k=50)
