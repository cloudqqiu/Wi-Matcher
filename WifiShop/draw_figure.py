import matplotlib.pyplot as plt
import numpy as np


def draw_qr():
    # x = [10, 100, 500, 1000, 2000, 3500, 5000]
    x = [i for i in range(1, 7)] #tkde
    y = [0.89915,
0.90544,
0.90983,
0.90796,
0.91044,
0.90839
    ]
    for index, i in enumerate(y):
        y[index] = i * 100

    plt.figure(figsize=(13.2, 10))
    plt.plot(x, y, '.-', linewidth=2, color='red')
    plt.scatter(x, y, edgecolor='red', s=30)
    plt.grid()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    plt.xlabel('Number of query recommendation entries', font2)
    plt.ylabel('$\mathregular{F_1}$(%)', font2)
    # plt.title('Err_rate for various k in kNN')
    plt.tick_params(labelsize=26)
    plt.xlim(1, 6)

    plt.ylim(89.5, 91.5)


    plt.show()


def draw_sr():
    x = [i for i in range(1, 11)]
    y = [0.8606,
0.90569,
0.9046,
0.90454,
0.90882,
0.90725,
0.90983,
0.91023,
0.90608,
0.90719
]
    for index, i in enumerate(y):
        y[index] = i * 100

    plt.figure(figsize=(14, 10))
    plt.plot(x, y, '.-', linewidth=2, color='red')
    plt.scatter(x, y, edgecolor='red', s=30)
    plt.grid()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    plt.xlabel('Number of search result entries', font2)
    plt.ylabel('$\mathregular{F_1}$(%)', font2)
    # plt.title('Err_rate for various k in kNN')
    plt.tick_params(labelsize=26)
    plt.xlim(1, 10)
    plt.ylim(86.0, 92.0)
    plt.show()


def draw_ngram():
    x = [i for i in range(1, 6)]
    y = [0.89366, 0.90343, 0.90983, 0.90894, 0.91051]
    for index, i in enumerate(y):
        y[index] = i * 100

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, '.-', linewidth=2, color='red')
    plt.scatter(x, y, edgecolor='red', s=30)
    plt.grid()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    plt.xlabel('N-gram', font2)
    plt.ylabel('$\mathregular{F_1}$(%)', font2)
    # plt.title('Err_rate for various k in kNN')
    plt.tick_params(labelsize=26)
    plt.xlim(1, 5)
    plt.xticks(x)
    plt.ylim(89, 91.5)
    plt.show()

if __name__ == '__main__':
    # draw_qr()
    draw_sr()

    # draw_ngram()

