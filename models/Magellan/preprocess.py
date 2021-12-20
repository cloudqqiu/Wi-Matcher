import pandas as pd
import models.model_config as config
import models.common.utils as utils

def ssid2org_magellan_format():
    data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
    data = data[['hotspot_id', 'ssid', 'venue_id', 'names', 'target']]
    data['names'] = data.apply(utils.select_first_name, axis=1)
    data.rename(columns={'hotspot_id': 'ssid_id'}, inplace=True)
    data.rename(columns={'target': 'label'}, inplace=True)
    data.index.name = 'index'

    ssid_data = data[['ssid_id', 'ssid']].drop_duplicates()
    ssid_data.rename(columns={'ssid_id': 'id'}, inplace=True)
    ssid_data.rename(columns={'ssid': 'name'}, inplace=True)

    venue_data = data[['venue_id', 'names']].drop_duplicates()
    venue_data.rename(columns={'venue_id': 'id'}, inplace=True)
    venue_data.rename(columns={'names': 'name'}, inplace=True)

    data.to_csv(f'{config.ru_magellan_data_path}/magellan_match.csv', index=1)
    ssid_data.to_csv(f'{config.ru_magellan_data_path}/magellan_wifi.csv', index=0)
    venue_data.to_csv(f'{config.ru_magellan_data_path}/magellan_shop.csv', index=0)


if __name__=='__main__':
    ssid2org_magellan_format()