import csv


def read_predict_result(pre_result):
    with open(pre_result, 'r') as pr:
        reader = csv.reader(pr)
        header = next(reader)
        data = [row for row in reader]
    pre_true_all = dict()
    for row in data:
        if float(row[1]) >= 0.5:
            amap_id, dp_id = row[0].split('.')
            if pre_true_all.__contains__(amap_id):
                pre_true_all[amap_id][dp_id] = row[1]
            else:
                temp = dict()
                temp[dp_id] = row[1]
                pre_true_all[amap_id] = temp
    del data
    pre_true = dict()
    for amap_id in pre_true_all.keys():
        max_dp_id = max(pre_true_all[amap_id], key=pre_true_all[amap_id].get)
        pre_true[amap_id] = max_dp_id
    return pre_true


def read_labeled(labeled):
    labeled_true = dict()
    with open(labeled, 'r') as pr:
        reader = csv.reader(pr)
        header = next(reader)
        data = [row for row in reader]
        for row in data:
            if row[1] == '1':
                labeled_true[row[0]] = 1
    return labeled_true


def cal_pre(pre_result, labeled):
    pre_true = read_predict_result(pre_result)
    labeled_true = read_labeled(labeled)

    correct = 0
    for amap_id in pre_true.keys():
        if labeled_true.__contains__('{}.{}'.format(amap_id, pre_true[amap_id])):
            correct += 1

    print('precision = {} / {} = {}'.format(correct, len(pre_true), correct / len(pre_true)))


if __name__ == '__main__':
    cal_pre('./src/experiment/hybrid_m_test.csv', './src/experiment/m_test.csv')