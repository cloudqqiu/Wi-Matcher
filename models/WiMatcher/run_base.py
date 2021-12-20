import argparse
import gc
import os
import sys
import time
from timeit import default_timer as timer

sys.path.append(os.getcwd()+"/../..")

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from data import *
from model import wi_matcher_base


def train_base(args):
    if args.dataset == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    elif args.dataset == 'ru':
        model_save_path = config.ru_wimatcher_data_path

    start_time_data_preprocess = timer()

    columns = ['ssid', 'venue']
    new_data, embed_mat_s, gram_len_s, char_len_s = extract_data_simple(args.dataset, process_column=columns, ngram=3)

    end_time_data_preprocess = timer()
    print(f"# Timer - Data Preprocess1 - {end_time_data_preprocess - start_time_data_preprocess}")

    pre_result = list()

    k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True)
    for fold_num, (train_index, test_index) in enumerate(k_fold.split(new_data, new_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, args.folds))
        start_time_data_preprocess = timer()

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

            end_time_data_preprocess = timer()
            print(f"# Timer - Data Preprocess2 - {end_time_data_preprocess - start_time_data_preprocess}")

            model = wi_matcher_base(
                args.max_seq_len, char_len_s, gram_len_s, embed_mat_s,
                args.nn_dim, args.num_dense, args.lr
            )

            time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
            model_name = 'base' + time_str

            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{model_name}.h5'

            start_time_model_train = timer()

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

            end_time_model_train = timer()
            print(f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

            model.load_weights(model_checkpoint_path)
            # test_result = model.evaluate([test[c] for c in columns], test_label, batch_size=BATCH_SIZE, verbose=1)
            # print(test_result)
            # pre_result.append(test_result)
            start_time_model_predict = timer()

            test_predict = model.predict([test[c] for c in columns], batch_size=args.batch_size, verbose=1)
            end_time_model_predict = timer()
            print(
                f"# Timer - Model Predict - Samples: {len(test)} Batch Size: {args.predict_batch_size} - {end_time_model_predict - start_time_model_predict}")

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
    parser.add_argument('--repeat_run', type=int, default=1)
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
        temp_p, temp_r = train_base(args)
        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print(f"Run 5-Fold for {times} times ")
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
