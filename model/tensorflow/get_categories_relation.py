from requests import get
import numpy as np

sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"

def sss(s1, s2, type='concept', corpus='webbase'):
    response = get(sss_url, params={'operation':'api','phrase1':s1,'phrase2':s2,'type':type,'corpus':corpus})
    return float(response.text.strip())


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
