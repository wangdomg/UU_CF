#!Python3
# -*- coding:utf-8 -*-
import pickle
import time

if __name__ == '__main__':
    start = time.asctime(time.localtime(time.time()))
    print('开始时间')
    print(start)

    print('读取测试数据')
    path = './data_L/user_click_test.txt'
    
    # 读取测试数据
    with open(path, 'r') as f:
        fobj = f.readlines()

        # user-news set，记录某个user所有浏览过的news的id
        test_user_news = {}

        # 创建user-news矩阵，只能包含后1/3的数据
        for i in range(len(fobj)):
            field_list = fobj[i].split()

            userid = field_list[0]
            newsid = field_list[1]

            # 记录每个user点击news的集合
            if userid in test_user_news:
                test_user_news[userid].add(newsid)
            else:
                test_user_news[userid] = set()
                test_user_news[userid].add(newsid)
    
    # 读取pkl
    print('读取pkl')
    with open('userid_list.pkl', 'rb') as f:
        userid_list = pickle.load(f)
    with open('newsid_list.pkl', 'rb') as f:
        newsid_list = pickle.load(f)
    with open('rnews.pkl', 'rb') as f:
        rnews = pickle.load(f)

    print('在测试集上测试准确率')
    # 在测试数据上测试准确率
    # 准确率=user从推荐中选择的/推荐的总数
    total_precison = 0.0
    total_user = 0
    for userid,news_set in test_user_news.items():
        # total_user += 1
        if userid in userid_list:  # 那些出现在前面训练集中的user 
            total_user += 1
            index = userid_list.index(userid)
            trnews = rnews[index]  # trnews长这个样子的 {newsindex:rating}
            # 将推荐的新闻按照得分降序排列
            sorted_trnews = sorted(trnews.items(), key=lambda d:d[1], reverse=True)
            # 取排名前6的新闻进行推荐，但是这里只是news的index
            trnews_index_set = set()
            i = 0
            for item in sorted_trnews:
                trnews_index_set.add(item[0])
                i += 1
                if i>= 5:
                    break
            # 根据news的index找到对应的newsid
            trnews_set = set()
            for index in trnews_index_set:
                trnews_set.add(newsid_list[index])
            # 计算准确度
            precision = len(news_set & trnews_set) / float(len(trnews_set))
            # 加到总的准确率上，所以最后是宏平均
            total_precison += precision
        else:  # 如果user未出现在前面的训练集中，就认为其precision为0.0
            total_precison += 0.0



    print('最终的准确率')
    # 打印准确率
    final_precision = total_precison/total_user
    print (final_precision)

    end = time.asctime(time.localtime(time.time()))
    print('结束时间')
    print(end)


    