import argparse
import gc
import os
import time
import sys
sys.path.append(os.getcwd()+"/../..")

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from tensorflow.python import debug as tf_debug
import tensorflow as tf
from timeit import default_timer as timer

from data import *
from model import wi_matcher_whole, wi_matcher_base, wi_matcher_whole_new


def train_base(args):
    if args.dataset == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    elif args.dataset == 'ru':
        model_save_path = config.ru_wimatcher_data_path

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    columns = ['ssid', 'venue']
    new_data, embed_mat_s, gram_len_s, char_len_s = extract_data_simple(args.dataset, process_column=columns, ngram=3)
    pre_result = list()

    k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True)
    for fold_num, (train_index, test_index) in enumerate(k_fold.split(new_data, new_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, args.folds))
        new_data_train = new_data.iloc[train_index]

        val_folder = StratifiedKFold(n_splits=10, shuffle=True)
        for t_index, val_index in val_folder.split(new_data_train, new_data_train['label']):
            # print(t_index, val_index)
            train, test, val = dict(), dict(), dict()
            for c in columns:
                tra = np.array(new_data_train.iloc[t_index][c])
                tra = sequence.pad_sequences(tra, maxlen=args.max_seq_len, padding='post')
                tes = np.array(new_data.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=args.max_seq_len, padding='post')
                va = np.array(new_data_train.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=args.max_seq_len, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va
            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']

            model = wi_matcher_base(
                args.max_seq_len, char_len_s, gram_len_s, embed_mat_s,
                args.nn_dim, args.num_dense, args.lr
            )

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
            model_name = 'base' + time_str

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'
            model.fit([train[c] for c in columns], train_label,
                      batch_size=args.batch_size,
                      epochs=args.max_epochs,
                      verbose=2,
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns], val_label),
                      callbacks=[
                          EarlyStopping(
                              monitor='val_loss',
                              min_delta=0.0001,
                              patience=3,
                              verbose=2,
                              mode='auto',
                          ),
                          # Save the weights of the best epoch.
                          ModelCheckpoint(
                              model_checkpoint_path,
                              monitor='val_loss',
                              save_best_only=True,
                              verbose=2,
                          ),
                      ])
            model.load_weights(model_checkpoint_path)
            # test_result = model.evaluate([test[c] for c in columns], test_label, batch_size=BATCH_SIZE, verbose=1)
            # print(test_result)
            # pre_result.append(test_result)
            test_predict = model.predict([test[c] for c in columns], batch_size=args.batch_size, verbose=1)
            t_label = test_label.values
            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if t_label[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if t_label[index] == 1:
                        fn += 1
            print(tp, fp, fn)
            try:
                print(tp / (tp + fp), tp / (tp + fn))
            except Exception as e:
                print(e)
            pre_result.append([tp, fp, fn])

            # return
            K.clear_session()
            del train, test, train_label, test_label
            del model
            gc.collect()
            break
    tp, fp, fn = 0, 0, 0
    for result in pre_result:
        tp += result[0]
        fp += result[1]
        fn += result[2]
    tp /= len(pre_result)
    fp /= len(pre_result)
    fn /= len(pre_result)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)
    return precision, recall


def train_full(args):
    # w-s ws-s-sr s-rec 三个模块结合 方法使用complexv2 和simplev5
    if args.dataset == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    elif args.dataset == 'ru':
        model_save_path = config.ru_wimatcher_data_path

    columns = ['ssid', 'venue']
    new_data_s, rec_result, gram_len_s, char_len_s, embed_mat_s = extract_data_simple_rec_v2(args.dataset,
                                                                                             max_rec_num=args.max_qr_num,
                                                                                             max_seq_len=args.max_seq_len,
                                                                                             process_column=columns,
                                                                                             ngram=5, fuzzy_rec=False)
    new_data, ws_data, search_result, max_s_seq_len, gram_len_c, char_len_c, embed_mat_c = extract_data_complex(
        args.dataset, process_column=columns, max_s_seq_len=args.max_s_seq_len, max_sr_num=args.max_sr_num, ngram=5,
        need_rec_score=False)
    pre_result = list()

    print('Data loaded')

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # tf_config = tf.ConfigProto(device_count={'GPU': 0})
    # sess = tf.Session(config=tf_config)
    # K.set_session(sess)
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

    k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True)
    for fold_num, (train_index, test_index) in enumerate(k_fold.split(new_data, new_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, args.folds))
        new_data_train = new_data.iloc[train_index]
        new_data_s_train = new_data_s.iloc[train_index]

        val_folder = StratifiedKFold(n_splits=10, shuffle=True)
        for t_index, val_index in val_folder.split(new_data_train, new_data_train['label']):
            # print(t_index, val_index)
            train, test, val = dict(), dict(), dict()
            for c in columns:
                tra = np.array(new_data_s_train.iloc[t_index][c])
                tra = sequence.pad_sequences(tra, maxlen=args.max_seq_len, padding='post')
                tes = np.array(new_data_s.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=args.max_seq_len, padding='post')
                va = np.array(new_data_s_train.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=args.max_seq_len, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va

            train_ws = np.array(ws_data.iloc[train_index].iloc[t_index]['ssid'])
            train_ws = sequence.pad_sequences(train_ws, maxlen=max_s_seq_len, padding='post')
            test_ws = np.array(ws_data.iloc[test_index]['ssid'])
            test_ws = sequence.pad_sequences(test_ws, maxlen=max_s_seq_len, padding='post')
            val_ws = np.array(ws_data.iloc[train_index].iloc[val_index]['ssid'])
            val_ws = sequence.pad_sequences(val_ws, maxlen=max_s_seq_len, padding='post')

            train_s = np.array(new_data_train.iloc[t_index]['venue'])
            train_s = sequence.pad_sequences(train_s, maxlen=max_s_seq_len, padding='post')
            test_s = np.array(new_data.iloc[test_index]['venue'])
            test_s = sequence.pad_sequences(test_s, maxlen=max_s_seq_len, padding='post')
            val_s = np.array(new_data_train.iloc[val_index]['venue'])
            val_s = sequence.pad_sequences(val_s, maxlen=max_s_seq_len, padding='post')

            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']
            train_sr = get_sr_respectively(new_data_train.iloc[t_index].copy(), search_result,
                                           min_s_seq_len=args.min_s_seq_len, max_s_seq_len=max_s_seq_len,
                                           max_sr_num=args.max_sr_num, gram_len_c=gram_len_c)
            test_sr = get_sr_respectively(new_data.iloc[test_index].copy(), search_result,
                                          min_s_seq_len=args.min_s_seq_len, max_s_seq_len=max_s_seq_len,
                                          max_sr_num=args.max_sr_num, gram_len_c=gram_len_c)
            val_sr = get_sr_respectively(new_data_train.iloc[val_index].copy(), search_result,
                                         min_s_seq_len=args.min_s_seq_len, max_s_seq_len=max_s_seq_len,
                                         max_sr_num=args.max_sr_num, gram_len_c=gram_len_c)
            train_rec, test_rec, val_rec = [0 for _ in range(args.max_qr_num)], [0 for _ in range(args.max_qr_num)], \
                                           [0 for _ in range(args.max_qr_num)]
            for i in range(args.max_qr_num):
                train_rec[i] = rec_result[i][train_index][t_index]
                test_rec[i] = rec_result[i][test_index]
                val_rec[i] = rec_result[i][train_index][val_index]

            model = wi_matcher_whole(
                args.max_seq_len,
                max_s_seq_len, char_len_c, gram_len_c, char_len_s,
                gram_len_s, embed_mat_c, embed_mat_s,
                args.nn_dim, args.num_dense, args.num_sec_dense
            )  # combine_model_v1  combine_dm_hy

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
            model_name = 'combine_v1' + time_str

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'
            model.fit([train[c] for c in columns] + [train_s, train_ws] + train_sr + train_rec, train_label,
                      batch_size=args.batch_size,
                      epochs=args.max_epochs,
                      verbose=2,  # 2 1
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + [val_s, val_ws] + val_sr + val_rec, val_label),
                      callbacks=[
                          EarlyStopping(
                              monitor='val_loss',
                              min_delta=0.0001,
                              patience=3,
                              verbose=2,
                              mode='auto',
                          ),
                          # Save the weights of the best epoch.
                          ModelCheckpoint(
                              model_checkpoint_path,
                              monitor='val_loss',
                              save_best_only=True,
                              verbose=2,
                          ),
                      ])
            model.load_weights(model_checkpoint_path)
            test_predict = model.predict([test[c] for c in columns] + [test_s, test_ws] + test_sr + test_rec,
                                         batch_size=args.batch_size, verbose=1)
            t_label = test_label.values
            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if t_label[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if t_label[index] == 1:
                        fn += 1
            print(tp, fp, fn)
            try:
                print(tp / (tp + fp), tp / (tp + fn))
            except Exception as e:
                print(e)
            pre_result.append([tp, fp, fn])

            # return
            K.clear_session()
            del train, test, train_label, test_label
            del model
            gc.collect()
            break
    tp, fp, fn = 0, 0, 0
    for result in pre_result:
        tp += result[0]
        fp += result[1]
        fn += result[2]
    tp /= len(pre_result)
    fp /= len(pre_result)
    fn /= len(pre_result)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)
    return precision, recall


def wi_matcher_dataload_new(args):
    if args.dataset == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    elif args.dataset == 'ru':
        model_save_path = config.ru_wimatcher_data_path

    base_data, label, search_result, rec_result = load_data(args.dataset)
    np_ssid, np_venue, np_qr_data, np_sr_venue, np_sr_sv, np_sr_data, qr_gram_len, qr_char_len, qr_embed_mat, sr_gram_len, sr_char_len, max_sr_seq_len, sr_embed_mat = encode_data(
        args.dataset, base_data, search_result, rec_result, ngram=args.ngram, qr_reorder=True, sr_reorder= True)
    label = label.to_numpy()

    kfold_result = []

    k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True)
    for fold_num, (train_val_index, test_index) in enumerate(k_fold.split(np_ssid, label)):
        print('Fold {} of {}\n'.format(fold_num + 1, args.folds))
        np_ssid_train_val, np_ssid_test = np_ssid[train_val_index], np_ssid[test_index]
        np_venue_train_val, np_venue_test = np_venue[train_val_index], np_venue[test_index]
        np_qr_data_train_val, np_qr_data_test = np_qr_data[train_val_index], np_qr_data[test_index]
        np_sr_venue_train_val, np_sr_venue_test = np_sr_venue[train_val_index], np_sr_venue[test_index]
        np_sr_sv_train_val, np_sr_sv_test = np_sr_sv[train_val_index], np_sr_sv[test_index]
        np_sr_data_train_val, np_sr_data_test = np_sr_data[train_val_index], np_sr_data[test_index]
        label_train_val, label_test = label[train_val_index], label[test_index]
        train_val_split = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, val_index in train_val_split.split(np_ssid_train_val, label_train_val):
            np_ssid_train, np_ssid_val = np_ssid[train_index], np_ssid[val_index]
            np_venue_train, np_venue_val = np_venue[train_index], np_venue[val_index]
            np_qr_data_train, np_qr_data_val = np_qr_data[train_index], np_qr_data[val_index]
            np_sr_venue_train, np_sr_venue_val = np_sr_venue[train_index], np_sr_venue[val_index]
            np_sr_sv_train, np_sr_sv_val = np_sr_sv[train_index], np_sr_sv[val_index]
            np_sr_data_train, np_sr_data_val = np_sr_data[train_index], np_sr_data[val_index]
            label_train, label_val = label[train_index], label[val_index]

            model = wi_matcher_whole(
                args.max_seq_len, max_sr_seq_len,
                sr_char_len, sr_gram_len, qr_char_len, qr_gram_len,
                sr_embed_mat, qr_embed_mat,
                args.nn_dim, args.num_dense, args.num_sec_dense
            )

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
            model_name = 'combine_v1' + time_str

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'
            model.fit([
                np_ssid_train,
                np_venue_train,
                np_sr_venue_train,
                np_sr_sv_train,
            ] + [np.squeeze(sr)for sr in np.hsplit(np_sr_data_train, args.max_sr_num)]+[np.squeeze(qr)for qr in np.hsplit(np_qr_data_train, args.max_qr_num)] ,
                label_train,
                batch_size=args.batch_size,
                epochs=args.max_epochs,
                verbose=2,
                validation_data=([
                    np_ssid_val,
                    np_venue_val,
                    np_sr_venue_val,
                    np_sr_sv_val,
                 ] + [np.squeeze(sr) for sr in np.hsplit(np_sr_data_val, args.max_sr_num)] + [np.squeeze(qr) for qr in np.hsplit(np_qr_data_val, args.max_qr_num)], label_val),
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        min_delta=0.0001,
                        patience=3,
                        verbose=2,
                        mode='auto',
                    ),
                    # Save the weights of the best epoch.
                    ModelCheckpoint(
                        model_checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=2,
                    ),
                ])
            model.load_weights(model_checkpoint_path)
            test_predict = model.predict([
                np_ssid_test,
                np_venue_test,
                np_sr_venue_test,
                np_sr_sv_test,
             ] + [np.squeeze(sr) for sr in np.hsplit(np_sr_data_test, args.max_sr_num)] + [np.squeeze(qr) for qr in np.hsplit(np_qr_data_test, args.max_qr_num)],
                batch_size=args.batch_size, verbose=1)
            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if label_test[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if label_test[index] == 1:
                        fn += 1
            print(tp, fp, fn)
            try:
                print(tp / (tp + fp), tp / (tp + fn))
            except Exception as e:
                print(e)
            kfold_result.append([tp, fp, fn])

            # return
            K.clear_session()
            del model
            gc.collect()
            break
    tp, fp, fn = 0, 0, 0
    for result in kfold_result:
        tp += result[0]
        fp += result[1]
        fn += result[2]
    tp /= len(kfold_result)
    fp /= len(kfold_result)
    fn /= len(kfold_result)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)
    return precision, recall

def wi_matcher_all_new(args):
    if args.dataset == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    elif args.dataset == 'ru':
        model_save_path = config.ru_wimatcher_data_path

    start_time_data_preprocess = timer()

    base_data, label, search_result, rec_result = load_data(args.dataset)
    np_ssid, np_venue, np_qr_data, np_sr_venue, np_sr_sv, np_sr_data, qr_gram_len, qr_char_len, qr_embed_mat, sr_gram_len, sr_char_len, max_sr_seq_len, sr_embed_mat = \
        encode_data(args.dataset, base_data, search_result, rec_result, ngram=args.ngram, max_qr_num=args.max_qr_num, max_sr_num=args.max_sr_num, qr_reorder=True, sr_reorder= False, remove_char=args.remove_char)
    label = label.to_numpy()

    end_time_data_preprocess = timer()
    print(f"# Timer - Data Preprocess - {end_time_data_preprocess - start_time_data_preprocess}")

    kfold_result = []

    k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True)
    for fold_num, (train_val_index, test_index) in enumerate(k_fold.split(np_ssid, label)):
        print('Fold {} of {}\n'.format(fold_num + 1, args.folds))
        np_ssid_train_val, np_ssid_test = np_ssid[train_val_index], np_ssid[test_index]
        np_venue_train_val, np_venue_test = np_venue[train_val_index], np_venue[test_index]
        np_qr_data_train_val, np_qr_data_test = np_qr_data[train_val_index], np_qr_data[test_index]
        np_sr_venue_train_val, np_sr_venue_test = np_sr_venue[train_val_index], np_sr_venue[test_index]
        np_sr_sv_train_val, np_sr_sv_test = np_sr_sv[train_val_index], np_sr_sv[test_index]
        np_sr_data_train_val, np_sr_data_test = np_sr_data[train_val_index], np_sr_data[test_index]
        label_train_val, label_test = label[train_val_index], label[test_index]
        train_val_split = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, val_index in train_val_split.split(np_ssid_train_val, label_train_val):
            np_ssid_train, np_ssid_val = np_ssid[train_index], np_ssid[val_index]
            np_venue_train, np_venue_val = np_venue[train_index], np_venue[val_index]
            np_qr_data_train, np_qr_data_val = np_qr_data[train_index], np_qr_data[val_index]
            np_sr_venue_train, np_sr_venue_val = np_sr_venue[train_index], np_sr_venue[val_index]
            np_sr_sv_train, np_sr_sv_val = np_sr_sv[train_index], np_sr_sv[val_index]
            np_sr_data_train, np_sr_data_val = np_sr_data[train_index], np_sr_data[val_index]
            label_train, label_val = label[train_index], label[val_index]

            model = wi_matcher_whole_new(
                args.max_seq_len, max_sr_seq_len, args.max_qr_num, args.max_sr_num,
                sr_char_len, sr_gram_len, qr_char_len, qr_gram_len,
                sr_embed_mat, qr_embed_mat,
                args.nn_dim, args.num_dense, args.num_sec_dense
            )

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
            model_name = 'combine_v1' + time_str

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'

            start_time_model_train = timer()

            model.fit([
                np_ssid_train,
                np_venue_train,
                np_sr_venue_train,
                np_sr_sv_train,
                np_sr_data_train,
                np_qr_data_train
            ] ,
                label_train,
                batch_size=args.batch_size,
                epochs=args.max_epochs,
                verbose=2,
                validation_data=([
                    np_ssid_val,
                    np_venue_val,
                    np_sr_venue_val,
                    np_sr_sv_val,
                    np_sr_data_val,
                    np_qr_data_val ], label_val),
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        min_delta=0.0001,
                        patience=3,
                        verbose=2,
                        mode='auto',
                    ),
                    # Save the weights of the best epoch.
                    ModelCheckpoint(
                        model_checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=2,
                    ),
                ])

            end_time_model_train = timer()
            print(f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

            model.load_weights(model_checkpoint_path)

            start_time_model_predict = timer()

            test_predict = model.predict([
                np_ssid_test,
                np_venue_test,
                np_sr_venue_test,
                np_sr_sv_test,
                np_sr_data_test,
                np_qr_data_test
            ],
                batch_size=args.predict_batch_size, verbose=1)

            end_time_model_predict = timer()
            print(f"# Timer - Model Predict - Samples: {len(label_test)} Batch Size: {args.predict_batch_size} - {end_time_model_predict - start_time_model_predict}")

            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if label_test[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if label_test[index] == 1:
                        fn += 1
            print(tp, '\t', fp, '\t', fn)
            try:
                print(tp / (tp + fp), tp / (tp + fn))
            except Exception as e:
                print(e)
            kfold_result.append([tp, fp, fn])

            # return
            K.clear_session()
            del model
            gc.collect()
            break
    tp, fp, fn = 0, 0, 0
    for result in kfold_result:
        tp += result[0]
        fp += result[1]
        fn += result[2]
    tp /= len(kfold_result)
    fp /= len(kfold_result)
    fn /= len(kfold_result)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)
    return precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wi-Matcher model")
    parser.add_argument('--dataset', '-d', required=True, choices=['ru', 'zh'], help='Dataset zh or ru')
    parser.add_argument('--max_qr_num', type=int, default=3)
    parser.add_argument('--max_sr_num', type=int, default=7)
    parser.add_argument('--rec_repeat', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--predict_batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--lr', type=int, default=None)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--max_s_seq_len', type=int, default=256)
    parser.add_argument('--min_s_seq_len', type=int, default=10)
    parser.add_argument('--nn_dim', type=int, default=300)
    parser.add_argument('--num_dense', type=int, default=128)
    parser.add_argument('--num_sec_dense', type=int, default=32)
    parser.add_argument('--save_log', default=False, type=bool)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--repeat_run', type=int, default=3)
    parser.add_argument('--remove_char', action='store_true')

    args = parser.parse_args()
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    # tf.config.experimental.list_physical_devices('GPU')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    session = tf.Session(config=config_tf)

    start_time = time.ctime()
    times = args.repeat_run
    _p, _r = 0, 0
    for i in range(times):
        # temp_p, temp_r = train_full(args)
        # temp_p, temp_r = train_base(args)
        # temp_p, temp_r = wi_matcher_dataload_new(args)
        temp_p, temp_r = wi_matcher_all_new(args)
        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print(f"Run 5-Fold for {times} times ")
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
