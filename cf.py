#!Python3
# -*- coding:utf-8 -*-
import numpy as np
import copy 
import time
import pickle

if __name__ == '__main__':
    begin = time.asctime(time.localtime(time.time()))
    print('开始时间')
    print(begin)

    path = './data_L/user_click_train.txt'
    
    print('读取训练数据')
    # 读取userid和newsid
    with open(path, 'r') as f:
        fobj = f.readlines()
        newsid_set = set()
        userid_set = set()

        # user-news set，记录某个user所有浏览过的news的id
        user_news_set = {}

        # 创建user-news矩阵
        for i in range(len(fobj)):
            field_list = fobj[i].split()

            userid = field_list[0]
            newsid = field_list[1]
            newsid_set.add(newsid)
            userid_set.add(userid)

            # 记录每个user点击news的集合
            if userid in user_news_set:
                user_news_set[userid].add(newsid)
            else:
                user_news_set[userid] = set()
                user_news_set[userid].add(newsid)

    print('构造user列表和news列表')
    # userid的有序列表
    userid_list = []
    for userid in userid_set:
        userid_list.append(userid)
    # newsid的有序列表
    newsid_list = []
    for newsid in newsid_set:
        newsid_list.append(newsid)
    print('用户数：', len(userid_list))
    print('新闻数：', len(newsid_list))

    print('初始化原始打分矩阵')
    # 原始的打分矩阵，用来计算相似度，预测user的打分，这个矩阵不会进行修改
    matrix = [[0 for col in range(len(newsid_list))] for row in range(len(userid_list))]

    print('填写原始打分矩阵')
    # 填写原始的打分矩阵，用户浏览过新闻则该位置为1，否则为0
    for i in range(len(userid_list)):
        userid = userid_list[i]
        for j in range(len(newsid_list)):
            if newsid_list[j] in user_news_set[userid]:
                matrix[i][j] = 1
    
    print('计算用户相似度')
    # 计算每两个user之间的similarity，暂时使用余弦相似度
    sim = {}
    for i in range(len(matrix)):
        sim[i] = {}
    mat = np.zeros([len(userid_list), len(newsid_list)])
    for i in range(len(matrix)):
        mat[i] = np.array(matrix[i])
    for i in range(len(mat)-1):
        print('用户id', i+1)
        for j in range(i+1, len(mat)):
            ui = mat[i]
            uj = mat[j]
            li = np.sqrt(ui.dot(ui))
            lj = np.sqrt(uj.dot(uj))
            s = ui.dot(uj)/(float(li)*lj)
            sim[i][j] = s
            sim[j][i] = s

    print('根据相似度填写打分矩阵')
    # 打分矩阵，用来记录所预测的user的打分，这个矩阵是不断进行修改的
    neighbors = {}  # 用来记录每个用户的邻居
    rating_matrix = copy.deepcopy(mat)
    for i in range(len(rating_matrix)):
        print('用户id：', i+1)
        sorted_sim_list = sorted(sim[i].items(), key=lambda d:d[1], reverse=True)  # 把跟user i相似的user按照similarity降序排列
        sim_userid_list = []
        num = 0
        user_id = userid_list[i]
        if user_id not in neighbors:
            neighbors[user_id] = []
        for item in sorted_sim_list:
            num += 1
            sim_userid_list.append(item[0])
            # 记录每个用户的邻居
            nei_user_id = userid_list[item[0]]
            neighbors[user_id].append(nei_user_id)
            if num >= 20:
                break
        for j in sim_userid_list:
            rating_matrix[i] += mat[j]*sim[i][j]
    

    print('记录为每个user推荐的新闻')
    rnews = {}
    for i in range(len(matrix)):
        rnews[i] = {}
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                rnews[i][j] = rating_matrix[i][j]

    # 保存数据
    with open('userid_list.pkl', 'wb') as f:
        pickle.dump(userid_list, f)
    with open('newsid_list.pkl', 'wb') as f:
        pickle.dump(newsid_list, f)
    with open('rnews.pkl', 'wb') as f:
        pickle.dump(rnews, f)
    with open('neighbors.pkl', 'wb') as f:
        pickle.dump(neighbors, f)

    end = time.asctime(time.localtime(time.time()))
    print('结束时间')
    print(end)