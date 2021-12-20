import argparse
import os
import sys
sys.path.append(os.getcwd()+"/../..")

import py_entitymatching as em
import numpy as np
import models.common.utils as utils
import models.model_config as config
import time
from timeit import default_timer as timer

def magellan(data):
    start_time_data_preprocess = timer()

    if data == 'zh':
        data_path = config.zh_magellan_data_path
        ssid_header = 'wifi'
        venue_header = 'match'
        tab_name_header = 'pinyin'
    elif data == 'ru':
        data_path = config.ru_magellan_data_path
        ssid_header = 'ssid_id'
        venue_header = 'venue_id'
        tab_name_header = 'name'

    match_tab_index_header = 'index'
    label_header = 'label'
    tab_index_header = 'id'

    print(f'Magellan Model dataset={data}')

    timer_log = utils.Timer()

    timer_log.start()
    w = em.read_csv_metadata(f'{data_path}/magellan_wifi.csv', key='id')
    s = em.read_csv_metadata(f'{data_path}/magellan_shop.csv', key='id')
    m = em.read_csv_metadata(f'{data_path}/magellan_match.csv', key='index')
    timer_log.log('Magellan Data loaded.')


    em.set_ltable(m, w)
    em.set_rtable(m, s)
    em.set_fk_ltable(m, ssid_header)
    em.set_fk_rtable(m, venue_header)

    end_time_data_preprocess = timer()
    print(f"# Timer - Data Preprocess 1 - {end_time_data_preprocess - start_time_data_preprocess}")

    ite_k = 10
    # avg_pre, avg_rec = 0, 0
    tp, fp, fn = 0, 0, 0
    for i in range(ite_k):
        timer_log.log(f'Magellan Run {i} start.')

        start_time_data_preprocess = timer()

        rng = np.random.RandomState(int(time.time()))
        rng.rand(4)
        IJ = em.split_train_test(m, train_proportion=0.7, random_state=rng)
        I = IJ['train']
        J = IJ['test']

        match_t = em.get_tokenizers_for_matching()
        match_s = em.get_sim_funs_for_matching()
        atypes1 = em.get_attr_types(w)  # don't need, if atypes1 exists from blocking step
        atypes2 = em.get_attr_types(s)  # don't need, if atypes2 exists from blocking step
        # print(atypes1, atypes2)
        atypes1[tab_index_header] = 'str_bt_1w_5w'
        # atypes1['name'] = 'str_bt_1w_5w'
        atypes1[tab_name_header] = 'str_bt_1w_5w'
        match_c = em.get_attr_corres(w, s)
        feature_table = em.get_features(w, s, atypes1, atypes2, match_c, match_t, match_s)
        # print(feature_table)
        # feature_table = em.get_features_for_matching(w, s, validate_inferred_attr_types=True)
        # return
        H = em.extract_feature_vecs(I,
                                    feature_table=feature_table,
                                    attrs_after=label_header,
                                    show_progress=False)

        L = em.extract_feature_vecs(J, feature_table=feature_table,
                                    attrs_after=label_header, show_progress=False)

        end_time_data_preprocess = timer()
        print(f"# Timer - Data Preprocess 2 - {end_time_data_preprocess - start_time_data_preprocess}")

        # magellan_choose_model(H)
        # magellan_model_ana(H, L)
        # return
        timer_log.log(f'Magellan Run {i} Feature generation')
        rf = em.RFMatcher(name='RF', random_state=0)

        start_time_model_train = timer()

        rf.fit(table=H, exclude_attrs=[match_tab_index_header, ssid_header, venue_header, label_header], target_attr=label_header)

        end_time_model_train = timer()
        print(f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

        timer_log.log(f'Magellan Run {i} Model Training')

        start_time_model_predict = timer()

        predictions = rf.predict(table=L, exclude_attrs=[match_tab_index_header, ssid_header, venue_header, label_header], append=True, target_attr='predicted', inplace=False)

        end_time_model_predict = timer()
        print(f"# Timer - Model Predict - {end_time_model_predict - start_time_model_predict}")

        timer_log.log(f'Magellan Run {i} Model Prediction')
        eval_result = em.eval_matches(predictions, label_header, 'predicted')
        tp += eval_result['prec_numerator']
        fp += eval_result['false_pos_num']
        fn += eval_result['false_neg_num']
        # em.print_eval_summary(eval_result)
        # avg_pre += eval_result['precision']
        # avg_rec += eval_result['recall']
    # avg_pre /= ite_k
    # avg_rec /= ite_k
    # print('Avg pre =', avg_pre)
    # print('Avg rec =', avg_rec)
    # print('Avg F1 =', 2 * avg_pre * avg_rec / (avg_pre + avg_rec))

    timer_log.stop(f'Magellan Model {ite_k} iterations finished.')

    tp /= ite_k
    fp /= ite_k
    fn /= ite_k
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Magellan model")
    parser.add_argument('--dataset', '-d', required=True, choices=['ru', 'zh'], help='Dataset zh or ru')

    args = parser.parse_args()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    magellan(data=args.dataset)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))