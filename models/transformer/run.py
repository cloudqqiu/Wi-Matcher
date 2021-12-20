import argparse
import gc
import os
import time
import sys

from keras.preprocessing import sequence

sys.path.append(os.getcwd()+"/../..")

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from tensorflow.python import debug as tf_debug
import tensorflow as tf
from timeit import default_timer as timer

from data import *
from model import transformer_model


def run_transformer_model(args):
    if args.dataset == 'zh':
        model_save_path = config.zh_transformer_path
    elif args.dataset == 'ru':
        model_save_path = config.ru_transformer_path

    start_time_data_preprocess = timer()

    columns = ['ssid', 'venue']
    np_ssid, np_venue, label, embed_mat_s, gram_len_s, char_len_s = extract_data_simple(args.dataset, args.max_seq_len, process_column=columns, ngram=3)
    pre_result = list()

    end_time_data_preprocess = timer()
    print(f"# Timer - Data Preprocess - {end_time_data_preprocess - start_time_data_preprocess}")

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

            model = transformer_model(
                args.max_seq_len, char_len_s, gram_len_s, embed_mat_s,
                nn_dim=args.nn_dim, num_dense=args.num_dense, lr=args.lr, num_layers=args.num_layers, num_heads=args.num_heads
            )

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
            model_name = 'base' + time_str

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'

            start_time_model_train = timer()

            model.fit([np_ssid_train, np_venue_train], label_train,
                      batch_size=args.batch_size,
                      epochs=args.max_epochs,
                      verbose=2,
                      # validation_split=0.1,
                      validation_data=([np_ssid_val, np_venue_val,], label_val),
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

            end_time_model_train = timer()
            print(f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

            model.load_weights(model_checkpoint_path)
            # test_result = model.evaluate([test[c] for c in columns], test_label, batch_size=BATCH_SIZE, verbose=1)
            # print(test_result)
            # pre_result.append(test_result)

            start_time_model_predict = timer()

            test_predict = model.predict([np_ssid_test, np_venue_test], batch_size=args.predict_batch_size, verbose=1)

            end_time_model_predict = timer()
            print(f"# Timer - Model Predict - Samples: {len(label_test)} Batch Size: {args.predict_batch_size} - {end_time_model_predict - start_time_model_predict}")

            t_label = label_test.values
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
    parser.add_argument('--lr', type=float, default=None)
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
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)

    args = parser.parse_args()
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    # tf.config.experimental.list_physical_devices('GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    start_time = time.ctime()
    times = args.repeat_run
    _p, _r = 0, 0
    for i in range(times):
        temp_p, temp_r = run_transformer_model(args)
        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print(f"Run 5-Fold for {times} times ")
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))