import pandas as pd
import models.model_config as config
import models.common.utils as utils
from sklearn.model_selection import StratifiedKFold


def deepmatcher_preprocess(data, folds=5):
    if data == 'ru':
        print(f'Splitting data for {data} dataset')
        data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
        data = data[['hotspot_id', 'ssid', 'venue_id', 'names', 'target']]
        data['names'] = data.apply(utils.select_first_name, axis=1)
        data.rename(columns={'hotspot_id': 'ssid_id'}, inplace=True)
        data.rename(columns={'ssid': 'ssid_text'}, inplace=True)
        data.rename(columns={'names': 'venue_text'}, inplace=True)
        data.rename(columns={'target': 'label'}, inplace=True)
        data.index.name = 'index'

        kf = StratifiedKFold(n_splits=folds, shuffle=True)
        k_index = 0
        for train_val_index, test_index in kf.split(data, data['label']):

            train_val_data = data.iloc[train_val_index]
            train_val_split = StratifiedKFold(n_splits=10, shuffle=True)
            for train_index, val_index in train_val_split.split(train_val_data, train_val_data['label']):
                break

            train_data = data.iloc[train_index]
            val_data = data.iloc[val_index]
            test_data = data.iloc[test_index]
            print("Fold", k_index, "TRAIN:", len(train_data), "VAL:", len(val_data), "TEST:", len(test_data))
            train_data.to_csv(f'{config.ru_deepmatcher_data_path}/dm_train_{k_index}.csv')
            val_data.to_csv(f'{config.ru_deepmatcher_data_path}/dm_val_{k_index}.csv')
            test_data.to_csv(f'{config.ru_deepmatcher_data_path}/dm_test_{k_index}.csv')
            k_index += 1


if __name__=='__main__':
    deepmatcher_preprocess(data='ru')