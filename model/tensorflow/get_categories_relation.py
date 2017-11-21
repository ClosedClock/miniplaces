from requests import get
import numpy as np
import matplotlib.pyplot as plt

sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"

def sss(s1, s2, type='concept', corpus='webbase'):
    response_string = ''
    while response_string == '':
        response = get(sss_url, params={'operation':'api','phrase1':s1,'phrase2':s2,'type':type,'corpus':corpus})
        response_string = response.text.strip()
        if response_string == '-Infinity':
            response_string = '0.0'
    print(s1, s2)
    print(response_string)
    return float(response_string)


def get_words_list():
    filename = '../../data/categories.txt'
    words_list = []
    with open(filename, 'r') as f:
        print('Files opens')
        for line in f.readlines():
            # print(line)
            word_with_number = line.split('/')[2]
            word = word_with_number.split(' ')[0]
            words_list.append(word.replace('_', ' '))
    return words_list


def save_relation_matrix():
    words_list = get_words_list()
    N = len(words_list)
    print('We get %d categories' % N)
    relation = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            word1 = words_list[i]
            word2 = words_list[j]
            relation[i][j] = sss(word1, word2)
            relation[j][i] = relation[i][j]
    np.savez('relation.npz', relation)
    return relation

def plot_heatmap(relation):
    print('Going to plot heatmap')
    words_list = get_words_list()

    fig, ax = plt.subplots()
    fig.canvas.draw()
    # ax.invert_yaxis()
    ax.set_xticklabels(words_list, fontsize=5)
    ax.set_yticklabels(words_list, fontsize=5)
    plt.xticks(np.arange(100))
    plt.yticks(np.arange(100))
    ax.xaxis.tick_top()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)


    cax = plt.imshow(relation, cmap='hot', interpolation='nearest')
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])

    plt.show()


relation = np.load('relation.npz')['arr_0']
plot_heatmap(relation)