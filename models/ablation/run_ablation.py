from model import wi_matcher_base_qr, wi_matcher_base_sr, hi_em_var, deepmatcher_var, deepmatcher_var_sr, \
    deepmatcher_var_qr, deepmatcher_var_qr_sr, hi_em_var_qr, hi_em_var_sr, hi_em_var_qr_sr
from data import *
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import argparse
import gc
import os
import time
import sys

sys.path.append(os.getcwd() + "/../..")


def wi_matcher_ablation(args):
    if args.dataset == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    elif args.dataset == 'ru':
        model_save_path = config.ru_wimatcher_data_path

    assert (args.qr == args.sr) == False

    start_time_data_preprocess = timer()

    base_data, label, search_result, rec_result = load_data(args.dataset)

    if args.qr:
        np_ssid, np_venue, np_qr_data, qr_gram_len, qr_char_len, qr_embed_mat = \
            encode_data_base_qr(args.dataset, base_data, rec_result, ngram=args.ngram,
                                max_qr_num=args.max_qr_num, qr_reorder=True,
                                remove_char=args.remove_char)
    elif args.sr:
        np_ssid, np_venue, np_sr_venue, np_sr_sv, np_sr_data, base_gram_len, base_char_len, base_embed_mat, sr_gram_len, sr_char_len, max_sr_seq_len, sr_embed_mat = \
            encode_data_base_sr(args.dataset, base_data, search_result, rec_result, ngram=args.ngram,
                                max_qr_num=args.max_qr_num, max_sr_num=args.max_sr_num, qr_reorder=True,
                                sr_reorder=False, remove_char=args.remove_char)

    label = label.to_numpy()

    end_time_data_preprocess = timer()
    print(
        f"# Timer - Data Preprocess - {end_time_data_preprocess - start_time_data_preprocess}")

    kfold_result = []

    k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True)
    for fold_num, (train_val_index, test_index) in enumerate(k_fold.split(np_ssid, label)):
        print('Fold {} of {}\n'.format(fold_num + 1, args.folds))
        np_ssid_train_val, np_ssid_test = np_ssid[train_val_index], np_ssid[test_index]
        np_venue_train_val, np_venue_test = np_venue[train_val_index], np_venue[test_index]
        if args.sr:
            np_sr_venue_train_val, np_sr_venue_test = np_sr_venue[
                train_val_index], np_sr_venue[test_index]
            np_sr_sv_train_val, np_sr_sv_test = np_sr_sv[train_val_index], np_sr_sv[test_index]
            np_sr_data_train_val, np_sr_data_test = np_sr_data[train_val_index], np_sr_data[test_index]
        elif args.qr:
            np_qr_data_train_val, np_qr_data_test = np_qr_data[train_val_index], np_qr_data[test_index]

        label_train_val, label_test = label[train_val_index], label[test_index]
        train_val_split = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, val_index in train_val_split.split(np_ssid_train_val, label_train_val):
            np_ssid_train, np_ssid_val = np_ssid[train_index], np_ssid[val_index]
            np_venue_train, np_venue_val = np_venue[train_index], np_venue[val_index]

            if args.sr:
                np_sr_venue_train, np_sr_venue_val = np_sr_venue[train_index], np_sr_venue[val_index]
                np_sr_sv_train, np_sr_sv_val = np_sr_sv[train_index], np_sr_sv[val_index]
                np_sr_data_train, np_sr_data_val = np_sr_data[train_index], np_sr_data[val_index]
            elif args.qr:
                np_qr_data_train, np_qr_data_val = np_qr_data[train_index], np_qr_data[val_index]

            label_train, label_val = label[train_index], label[val_index]

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())

            if args.sr:
                model = wi_matcher_base_sr(
                    args.max_seq_len, max_sr_seq_len, args.max_sr_num,
                    sr_char_len, sr_gram_len, base_char_len, base_gram_len,
                    sr_embed_mat, base_embed_mat,
                    args.nn_dim, args.num_dense, args.num_sec_dense
                )
                model_name = 'wimatcher_base_sr_' + time_str

            elif args.qr:
                model = wi_matcher_base_qr(
                    args.max_seq_len, args.max_qr_num,
                    qr_char_len, qr_gram_len, qr_embed_mat,
                    args.nn_dim, args.num_dense, args.num_sec_dense
                )
                model_name = 'wimatcher_base_qr_' + time_str

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'

            start_time_model_train = timer()

            if args.sr:
                model.fit([np_ssid_train, np_venue_train, np_sr_venue_train, np_sr_sv_train, np_sr_data_train],
                          label_train,
                          batch_size=args.batch_size, epochs=args.max_epochs, verbose=2,
                          validation_data=(
                              [np_ssid_val, np_venue_val, np_sr_venue_val, np_sr_sv_val, np_sr_data_val], label_val),
                          callbacks=[
                              EarlyStopping(
                                  monitor='val_loss', min_delta=0.0001, patience=3, verbose=2, mode='auto', ),
                              ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True,
                                              save_weights_only=True, verbose=2, ),
                ])
            elif args.qr:
                model.fit([np_ssid_train, np_venue_train, np_qr_data_train],
                          label_train,
                          batch_size=args.batch_size, epochs=args.max_epochs, verbose=2,
                          validation_data=(
                              [np_ssid_val, np_venue_val, np_qr_data_val], label_val),
                          callbacks=[
                              EarlyStopping(
                                  monitor='val_loss', min_delta=0.0001, patience=3, verbose=2, mode='auto', ),
                              # Save the weights of the best epoch.
                              ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True,
                                              save_weights_only=True, verbose=2, ),
                ])

            end_time_model_train = timer()
            print(
                f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

            model.load_weights(model_checkpoint_path)

            start_time_model_predict = timer()

            if args.sr:
                test_predict = model.predict([
                    np_ssid_test, np_venue_test, np_sr_venue_test, np_sr_sv_test, np_sr_data_test,
                ], batch_size=args.predict_batch_size, verbose=1)
            elif args.qr:
                test_predict = model.predict([
                    np_ssid_test, np_venue_test, np_qr_data_test
                ], batch_size=args.predict_batch_size, verbose=1)

            end_time_model_predict = timer()
            print(
                f"# Timer - Model Predict - Samples: {len(label_test)} Batch Size: {args.predict_batch_size} - {end_time_model_predict - start_time_model_predict}")

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


def hi_em_var_ablation(args):
    if args.dataset == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    else:
        raise Exception

    start_time_data_preprocess = timer()

    columns = ['ssid', 'venue']
    base_data, label, search_result, rec_result = load_data(
        args.dataset, qr=args.qr, sr=args.sr)

    np_ssid, np_venue, label, embed_mat_s, gram_len_s, char_len_s = load_base_data(args.dataset,
                                                                                   process_column=columns, ngram=3,
                                                                                   remove_char=args.remove_char)

    label = label.to_numpy()

    end_time_data_preprocess = timer()
    print(
        f"# Timer - Data Preprocess - {end_time_data_preprocess - start_time_data_preprocess}")

    kfold_result = []

    k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True)
    for fold_num, (train_val_index, test_index) in enumerate(k_fold.split(np_ssid, label)):
        print('Fold {} of {}\n'.format(fold_num + 1, args.folds))
        np_ssid_train_val, np_ssid_test = np_ssid[train_val_index], np_ssid[test_index]
        np_venue_train_val, np_venue_test = np_venue[train_val_index], np_venue[test_index]

        label_train_val, label_test = label[train_val_index], label[test_index]
        train_val_split = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, val_index in train_val_split.split(np_ssid_train_val, label_train_val):
            np_ssid_train, np_ssid_val = np_ssid[train_index], np_ssid[val_index]
            np_venue_train, np_venue_val = np_venue[train_index], np_venue[val_index]

            label_train, label_val = label[train_index], label[val_index]

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())

            model_name = 'HI_EM_var_' + time_str
            model = hi_em_var(args.max_seq_len, char_len_s, gram_len_s,
                              embed_mat_s, args.nn_dim, args.num_dense, args.lr)

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'

            start_time_model_train = timer()

            model.fit([np_ssid_train, np_venue_train], label_train,
                      batch_size=args.batch_size, epochs=args.max_epochs, verbose=2,
                      validation_data=([np_ssid_val, np_venue_val], label_val),
                      callbacks=[
                EarlyStopping(monitor='val_loss', min_delta=0.0001,
                              patience=3, verbose=2, mode='auto', ),
                ModelCheckpoint(model_checkpoint_path, monitor='val_loss',
                                save_best_only=True, verbose=2)
            ])

            end_time_model_train = timer()
            print(
                f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

            model.load_weights(model_checkpoint_path)

            start_time_model_predict = timer()

            test_predict = model.predict([np_ssid_test, np_venue_test], batch_size=args.predict_batch_size,
                                         verbose=1)

            end_time_model_predict = timer()
            print(
                f"# Timer - Model Predict - Samples: {len(label_test)} Batch Size: {args.predict_batch_size} - {end_time_model_predict - start_time_model_predict}")

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


def deepmatcher_var_ablation(args):
    if args.dataset == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    else:
        raise Exception

    start_time_data_preprocess = timer()

    columns = ['ssid', 'venue']

    base_data, label, search_result, rec_result = load_data(
        args.dataset, qr=args.qr, sr=args.sr)
    np_ssid, np_venue, label, embed_mat_s, gram_len_s, char_len_s = load_base_data(args.dataset,
                                                                                   process_column=columns, ngram=3,
                                                                                   remove_char=args.remove_char)

    label = label.to_numpy()

    end_time_data_preprocess = timer()
    print(
        f"# Timer - Data Preprocess - {end_time_data_preprocess - start_time_data_preprocess}")

    kfold_result = []

    k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True)
    for fold_num, (train_val_index, test_index) in enumerate(k_fold.split(np_ssid, label)):
        print('Fold {} of {}\n'.format(fold_num + 1, args.folds))
        np_ssid_train_val, np_ssid_test = np_ssid[train_val_index], np_ssid[test_index]
        np_venue_train_val, np_venue_test = np_venue[train_val_index], np_venue[test_index]

        label_train_val, label_test = label[train_val_index], label[test_index]
        train_val_split = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, val_index in train_val_split.split(np_ssid_train_val, label_train_val):
            np_ssid_train, np_ssid_val = np_ssid[train_index], np_ssid[val_index]
            np_venue_train, np_venue_val = np_venue[train_index], np_venue[val_index]

            label_train, label_val = label[train_index], label[val_index]

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())

            model_name = 'DeepMatcher_var_' + time_str
            model = deepmatcher_var(args.max_seq_len, char_len_s, gram_len_s, embed_mat_s, args.nn_dim,
                                    args.num_dense)

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'

            start_time_model_train = timer()

            model.fit([np_ssid_train, np_venue_train], label_train,
                      batch_size=args.batch_size, epochs=args.max_epochs, verbose=2,
                      validation_data=([np_ssid_val, np_venue_val], label_val),
                      callbacks=[
                EarlyStopping(monitor='val_loss', min_delta=0.0001,
                              patience=3, verbose=2, mode='auto', ),
                ModelCheckpoint(model_checkpoint_path, monitor='val_loss',
                                save_best_only=True, verbose=2)
            ])

            end_time_model_train = timer()
            print(
                f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

            model.load_weights(model_checkpoint_path)

            start_time_model_predict = timer()

            test_predict = model.predict([np_ssid_test, np_venue_test], batch_size=args.predict_batch_size,
                                         verbose=1)

            end_time_model_predict = timer()
            print(
                f"# Timer - Model Predict - Samples: {len(label_test)} Batch Size: {args.predict_batch_size} - {end_time_model_predict - start_time_model_predict}")

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
    parser = argparse.ArgumentParser(description="Wi-Matcher ablation model")
    parser.add_argument('--dataset', '-d', required=True,
                        choices=['ru', 'zh'], help='Dataset zh or ru')
    parser.add_argument('--model', required=True,
                        choices=['wimatcher', 'hi_em', 'deepmatcher'])
    parser.add_argument('--max_qr_num', type=int, default=3)
    parser.add_argument('--max_sr_num', type=int, default=7)
    parser.add_argument('--rec_repeat', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--predict_batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--max_s_seq_len', type=int, default=256)
    parser.add_argument('--min_s_seq_len', type=int, default=10)
    parser.add_argument('--nn_dim', type=int, default=300)
    parser.add_argument('--num_dense', type=int, default=128)
    parser.add_argument('--num_sec_dense', type=int, default=32)
    parser.add_argument('--save_log', default=False, type=bool)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--repeat_run', type=int, default=1)
    parser.add_argument('--remove_char', action='store_true')
    parser.add_argument('--sr', action='store_true')
    parser.add_argument('--qr', action='store_true')

    args = parser.parse_args()
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    # tf.config.experimental.list_physical_devices('GPU')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    models_dict = {
        'wimatcher': wi_matcher_ablation,
        'hi_em': hi_em_var_ablation,
        'deepmatcher': deepmatcher_var_ablation,
    }

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    session = tf.Session(config=config_tf)

    start_time = time.ctime()
    times = args.repeat_run
    _p, _r = 0, 0
    for i in range(times):
        temp_p, temp_r = models_dict[args.model](args)
        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print(f"Run 5-Fold for {times} times ")
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
