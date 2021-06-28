import csv

headers = ['id', 'label', 'left_Id', 'left_Name', 'left_Name_Full', 'left_Addr', 'left_Lat', 'left_Lng', 'right_Id',
           'right_Name', 'right_Name_Full', 'right_Addr', 'right_Lat', 'right_Lng']


def single_lat_lng(path, f):
    with open('{}/{}'.format(path, f), 'r') as c:
        c_reader = csv.reader(c)
        old_header = next(c_reader)
        data = [row for row in c_reader]
        for row in data:
            r_lat, r_lng = row[-1].split(',')
            l_lat, l_lng = row[5].split(',')
            row[-1] = r_lat
            row.append(r_lng)
            row[5] = l_lng
            row.insert(5, l_lat)
    with open('{}/{}'.format(path, f), 'w') as c:
        f_csv = csv.writer(c, lineterminator='\n')
        f_csv.writerow(headers)
        f_csv.writerows(data)


def add_id(path, f):
    with open('{}/{}'.format(path, f), 'r', encoding='utf-8') as c:
        c_reader = csv.reader(c)
        old_header = next(c_reader)
        data = [row for row in c_reader]
        for row in data:
            amap_id, dp_id = row[0].split('.')
            row.insert(7, dp_id)
            row.insert(2, amap_id)
    with open('{}/{}'.format(path, f), 'w') as c:
        f_csv = csv.writer(c, lineterminator='\n')
        f_csv.writerow(headers)
        f_csv.writerows(data)

if __name__ == '__main__':
    single_lat_lng('./src/experiment', 'm_test.csv')
    add_id('./src/experiment', 'm_test.csv')
