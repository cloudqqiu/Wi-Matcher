import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from collections import Counter
import WifiShop.data_process as dp
import pro_func as pf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

ex_path = dp.ex_path  # '../src/experiment'
ngram = dp.ngram  # 3



def process_data(pois, save=False):
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
            pos = pd.read_csv('{}/linking/prepro/pos_{}.csv'.format(ex_path, poi))
            neg = pd.read_csv('{}/linking/prepro/neg_{}.csv'.format(ex_path, poi))
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
        with open('{}/new dp/data/{}.txt'.format(shop_path, '-'.join([pf.get_rid_of_zero(_) for _ in poi.split(',')])), 'r', encoding='utf-8') as f:
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
            text_list_py.append(pf.chinese2pinyin(row['wifi']))
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
            temp_text_py = pf.chinese2pinyin(temp_text)
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
            temp_sample['ssid'] = pf.chinese2pinyin(row['wifi'])
            temp_sample['name'] = pf.chinese2pinyin(match_shop_text_list[index])
            temp_sample['label'] = row['label']
            temp_sample['ssid_len'] = len(row['wifi'])
            temp_sample['ssid_token_len'] = len(row['wifi'].split(' '))
            temp_sample['names_len_mean'] = len(temp_sample['name'])
            temp_sample['names_tokens_count_mean'] = len(temp_sample['name'].split(' '))
            temp_sample['name_char_trigram_cos'] = pf.cos_onehot_vector(
                pf.get_ngram(temp_sample['ssid'], 3), pf.get_ngram(temp_sample['name'], 3))
            temp_sample['name_char_trigram_cover'] = \
                len(set(pf.get_ngram(temp_sample['ssid'], 3)) & set(pf.get_ngram(temp_sample['name'], 3))) \
                / len(set(pf.get_ngram(temp_sample['name'], 3)))
            temp_sample['name_char_cos'] = pf.cos_onehot_vector(set(temp_sample['ssid']), set(temp_sample['name']))
            temp_sample['name_char_cover'] = len(set(temp_sample['ssid']) & set(temp_sample['name'])) \
                / len(set(temp_sample['name']))
            temp_sample['name_token_cos'] = pf.cos_onehot_vector(
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
        total_df.to_csv('{}/linking/matching/cikm19/dataset.csv'.format(ex_path))
    return total_df


def cikm19_gbdt(folds=5):
    try:
        df = pd.read_csv('{}/linking/matching/cikm19/dataset.csv'.format(ex_path))
    except Exception as e:
        import warnings
        warnings.warn(e.__str__())
    # print(1)
    # print(df.info())
    features = df.columns.tolist()
    f_features = np.where(df.dtypes == np.float)[0].tolist()
    f_f_name = [features[i] for i in f_features]
    df[f_f_name] = df[f_f_name].astype('str')
    # print(df.info())
    features.remove('Unnamed: 0')
    features.remove('label')
    features.remove('name')  ##
    features.remove('ssid')  ##
    new_df = df.drop(['label', 'name', 'ssid', 'Unnamed: 0'], axis=1) # name ssid
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    all_pred_result = list()
    for index, (train, test) in enumerate(skf.split(df, df['label'])):
        print('Fold: ', index)
        # print(train, test)

        skf_v = StratifiedKFold(n_splits=folds - 1, shuffle=True)
        trainwv, valwv = next(skf_v.split(df.iloc[train], df.iloc[train]['label']))

        model = CatBoostClassifier(iterations=2000, depth=9, cat_features=features, l2_leaf_reg=3,
                                   early_stopping_rounds=5,
                                   eval_metric='Logloss',
                                   learning_rate=0.05,
                                   loss_function='Logloss')
        # model.fit(new_df.iloc[train], df.iloc[train]['label'], plot=True)
        model.fit(new_df.iloc[train].iloc[trainwv], df.iloc[train].iloc[trainwv]['label'], plot=True,
                  eval_set=(new_df.iloc[train].iloc[valwv], df.iloc[train].iloc[valwv]['label']),
                  )
        print(model.tree_count_)

        # pred = model.predict(df.iloc[test], df.iloc[test]['label'])
        preds_class = model.predict(new_df.iloc[test])
        # preds_proba = model.predict_proba(df.iloc[test])
        print("class=", preds_class)
        # print("proba=", preds_proba)
        tp, fp, fn = 0, 0, 0
        for ind, i in enumerate(preds_class):
            # print(df.iloc[test].iloc[ind]['name'], df.iloc[test].iloc[ind]['ssid'],  i, df.iloc[test].iloc[ind]['label'])
            if i == 1:
                if df.iloc[test].iloc[ind]['label'] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if df.iloc[test].iloc[ind]['label'] == 1:
                    fn += 1
        print(tp, fp, fn)
        print(tp / (tp + fp), tp / (tp + fn))
        all_pred_result.append([tp, fp, fn])

        import matplotlib.pyplot as plt
        fea_ = model.feature_importances_
        fea_name = model.feature_names_
        plt.figure(figsize=(10, 10))
        plt.barh(fea_name, fea_, height=0.5)
        plt.show()
        break

    print(all_pred_result)
    tp, fp, fn = 0, 0, 0
    for result in all_pred_result:
        tp += result[0]
        fp += result[1]
        fn += result[2]
    tp /= len(all_pred_result)
    fp /= len(all_pred_result)
    fn /= len(all_pred_result)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)
    return precision, recall


if __name__ == '__main__':
    pois = ['39.92451,116.51533', '39.93483,116.45241',  # 这两个是第一批
            '39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
            '39.96333,116.45187', '39.98850,116.41674', '40.00034,116.46960']
    # process_data(pois, save=True)
    cikm19_gbdt(5)
