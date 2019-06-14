import numpy as np
import time
import os

import logging

from time import strftime, gmtime




Test_count = 100  #测试集大小
Train_count = 4000  #训练集大小
Classes = 5   #类别数
numOfTest = 100  #测试图片数
numOfAll = 4100  #全部图片数
#k = 60   #top-k的k 10 20 40 60 100 150
d = 5 #指定汉明距离
return_num = 200  #召回数
CURRENT_DIR = os.getcwd()


#HASHING_BITS  =32 #8 16 24# 32  48  64

#获取当前时间
def getNowTime():
    return '[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ']'

#从txt读取图片名
def Get_Names(Names):
    with open(CURRENT_DIR + '/juhua/new_testLabels.txt', 'r') as f:
        for line in f:
            temp1 = line.strip().split('\t')
            Names.append(temp1[0])
    f.close()
    with open(CURRENT_DIR + '/juhua/new_trainLabels.txt', 'r') as f:
        for line in f:
            temp2 = line.strip().split('\t')
            Names.append(temp2[0])
    f.close()
    logging.info('{}read Names finish'.format(getNowTime()))

#从txt读取二进制编码和标签信息
def Get_CodesLabels(train_codes, train_labels, test_codes, test_labels, lab,HASHING_BITS):
    line_number = 0
    file_res_name = CURRENT_DIR +"/result" + str(HASHING_BITS) + ".txt"
   # with open(CURRENT_DIR + '/result.txt', 'r') as f:
    with open(file_res_name, 'r') as f:
        for line in f:
            temp = line.strip().split('\t')
            if line_number < Test_count:
                test_codes.append(
                    [i if i == 1 else -1 for i in map(int, list(temp[0]))])
                list2 = [0] * Classes
                list2[int(temp[1])] = 1
                lab.append(temp[1])
                test_labels.append(list2)
            else:
                train_codes.append([i if i == 1 else -1 for i in map(int, list(temp[0]))])
                list2 = [0] * Classes
                list2[int(temp[1])] = 1
                lab.append(temp[1])
                train_labels.append(list2)
            line_number += 1
    logging.info('{}read data finish'.format(getNowTime()))



#计算汉明距离
def getHammingDist(code_a, code_b):
    dist = 0
    for i in range(len(code_a)):
        if code_a[i] != code_b[i]:
            dist += 1
    return dist

#toDo 在与查询图像的汉明距离最小的图像中正确结果所占的比例
#汉明距离最短的结果总数为n，其中正确个数为x
#x/n得到准确率，再求平均值
def Calculate_ByMinHanmming(HASHING_BITS):
    start = time.time()
    n = np.zeros((Classes, 1), np.int32)
    x = np.zeros((Classes, 1), np.int32)
    acc =np.zeros((Classes,1),np.float64)

    #####################

    train_codes = []  # 训练集二进制码
    train_labels = []  # 训练集标签数组
    test_codes = []  # 测试集二进制码
    test_labels = []  # 测试集标签数组
    codes = []  # 全部数据集二进制码
    labels = []  # 全部数据级标签数组
    lab = []  # 标签值
    Names = []  # 数据集图片名
    Get_Names(Names)
    # get g.t. and binary code
    Get_CodesLabels(train_codes, train_labels, test_codes, test_labels, lab,HASHING_BITS)
    codes.extend(test_codes)
    codes.extend(train_codes)
    labels.extend(test_labels)
    labels.extend(train_labels)
    codes = np.array(codes)
    labels = np.array(labels)
    train_codes = np.array(train_codes)  # 4000*24
    train_labels = np.array(train_labels)  # 4000*5
    test_codes = np.array(test_codes)  # 100*24
    test_labels = np.array(test_labels)  # 100*5

    # 生成汉明矩阵、标签矩阵
    gt_martix = np.dot(test_labels, np.transpose(labels))  # 100*4000
    logging.info('{}gt_martix finish!'.format(getNowTime()))
    ham_matrix = np.dot(test_codes, np.transpose(codes))  # hanmming distance map to dot value 100*4000
    logging.info('{}ham_martix finish!'.format(getNowTime()))

    # 汉明矩阵排序
    sorted_ham_martix = np.argsort(ham_matrix, axis=1)  #
    # logging.info(sorted_ham_martix)
    # logging.info(ham_martix[0][sorted_ham_martix[0]])
    # 计算准确率
    logging.info('{}sort ham_matrix finished,start calculate precision'.format(getNowTime()))

    p = np.zeros((numOfTest, 1), np.float64)  # 100*1

################
    logging.info('\n==================================1 Calculate By leatest MinHanmming==================================')
    for i in range(numOfTest):
        nn = 0.0
        xx = 0.0
        test_matrix = sorted_ham_martix[i, :]
        length = test_matrix.shape[0]
        distance = getHammingDist(codes[i], codes[test_matrix[length-1]])
     #   logging.info('第%d个查询为:%s'%(i+1, Names[i]))
      #  logging.info('\t与%d个查询最小的汉明距离is：%d'%(i+1, distance))
   #     logging.info('\t与第%d个查询汉明距离最小的样本为：'%(i+1))
        for j in range(numOfAll):
            if ham_matrix[i][test_matrix[length-j-1]] == ham_matrix[i][test_matrix[length-1]]:  #找到汉明距离最短的结果
                n[int(lab[i])] += 1
                nn += 1
                if gt_martix[i][test_matrix[length - j - 1]] == 1:  #正确的结果
                    x[int(lab[i])] += 1
                    xx += 1
                 #   logging.info('\t%s'%Names[test_matrix[length - j - 1]])
            else:
                break
       # logging.info('\t与第%d个样本汉明距离最小的个数为：%d'%(i, nn))
        p[i] = xx / nn
    logging.info('mean precision by MinHanmming is:%4f'%np.mean(p))
    for i in range(Classes):
        acc[i] = x[i] / n[i]
        logging.info('第%d类的Min_HanmmingDistance准确率为：%4f'%(i, acc[i]))

    end = time.time()
    logging.info('Average evaluate time is %4f s' % ((end - start) / numOfTest))
    accuracy = np.mean(acc)
    logging.info('mean precision by MinHanmming is:%4f'%accuracy)
    #return accuracy


#toDo 在与查询图像的汉明距离小于d 的图像中正确结果所占的比例
def Calculate_ByHanmming_d(HASHING_BITS):
    start = time.time()
    logging.info('\n==================================2 Calculate_ByHanmming_d==================================')
    n = np.zeros((Classes, 1), np.int32)
    x = np.zeros((Classes, 1), np.int32)
    acc = np.zeros((Classes, 1), np.float64)

    ##############################

    train_codes = []  # 训练集二进制码
    train_labels = []  # 训练集标签数组
    test_codes = []  # 测试集二进制码
    test_labels = []  # 测试集标签数组
    codes = []  # 全部数据集二进制码
    labels = []  # 全部数据级标签数组
    lab = []  # 标签值
    Names = []  # 数据集图片名
    Get_Names(Names)
    # get g.t. and binary code
    Get_CodesLabels(train_codes, train_labels, test_codes, test_labels, lab,HASHING_BITS)
    codes.extend(test_codes)
    codes.extend(train_codes)
    labels.extend(test_labels)
    labels.extend(train_labels)
    codes = np.array(codes)
    labels = np.array(labels)
    train_codes = np.array(train_codes)  # 4000*24
    train_labels = np.array(train_labels)  # 4000*5
    test_codes = np.array(test_codes)  # 100*24
    test_labels = np.array(test_labels)  # 100*5

    # 生成汉明矩阵、标签矩阵
    gt_martix = np.dot(test_labels, np.transpose(labels))  # 100*4000
    logging.info('{}gt_martix finish!'.format(getNowTime()))
    ham_matrix = np.dot(test_codes, np.transpose(codes))  # hanmming distance map to dot value 100*4000
    logging.info('{}ham_martix finish!'.format(getNowTime()))

    # 汉明矩阵排序
    sorted_ham_martix = np.argsort(ham_matrix, axis=1)  #
    # logging.info(sorted_ham_martix)
    # logging.info(ham_martix[0][sorted_ham_martix[0]])
    # 计算准确率
    logging.info('{}sort ham_matrix finished,start calculate precision'.format(getNowTime()))

    p = np.zeros((numOfTest, 1), np.float64)  # 100*1

##############################
    for i in range(numOfTest):
        nn = 0.0
        xx = 0.0
        test_matrix = sorted_ham_martix[i, :]
        length = test_matrix.shape[0]
       # logging.info('第%d个样本is:%s' % (i, Names[i]))
       # logging.info('\t与第%d个样本汉明距离小于%d的样本为：' %(i, d))
        for j in range(numOfAll):
            if getHammingDist(codes[i], codes[test_matrix[length - j - 1]]) <= d:
                n[int(lab[i])] += 1
                nn += 1
                if gt_martix[i][test_matrix[length - j - 1]] == 1:  # 正确的结果
                    x[int(lab[i])] += 1
                    xx += 1
             #   logging.info('\t%s'%Names[test_matrix[length - j - 1]])
            else:
                break
      #  logging.info('\t与第%d个样本汉明距离为%d的个数为：%d'%(i, d, nn))
        p[i] = xx / nn
    logging.info('mean precision by HanmmingDistance %d is:%4f' % (d, np.mean(p)))
    for i in range(Classes):
        acc[i] = x[i] / n[i]
        logging.info('第%d类的准确率 by Hanmming Dist:%d is：%4f' % (i, d, acc[i]))
    end = time.time()
    logging.info('Average evaluate time is %4f s' % ((end - start) / numOfTest))

    accuracy = np.mean(acc)
    logging.info('mean precision of MinHanmming is: %4f'%accuracy)
    # return accuracy


#toDo 与查询图像距离最小的k 张图像中正确结果所占的比例
def Calculate_Top_k(k,HASHING_BITS):
    start = time.time()
    logging.info('\n==============================3 Calculate Top k=================================='  )
    x = np.zeros((Classes, 1), np.int32)
    acc = np.zeros((Classes, 1), np.float64)
    count = np.zeros((Classes, 1), np.float64)
    ###########################################

    train_codes = []  # 训练集二进制码
    train_labels = []  # 训练集标签数组
    test_codes = []  # 测试集二进制码
    test_labels = []  # 测试集标签数组
    codes = []  # 全部数据集二进制码
    labels = []  # 全部数据级标签数组
    lab = []  # 标签值
    Names = []  # 数据集图片名
    Get_Names(Names)
    # get g.t. and binary code
    Get_CodesLabels(train_codes, train_labels, test_codes, test_labels, lab,HASHING_BITS)
    codes.extend(test_codes)
    codes.extend(train_codes)
    labels.extend(test_labels)
    labels.extend(train_labels)
    codes = np.array(codes)
    labels = np.array(labels)
    train_codes = np.array(train_codes)  # 4000*24
    train_labels = np.array(train_labels)  # 4000*5
    test_codes = np.array(test_codes)  # 100*24
    test_labels = np.array(test_labels)  # 100*5

    # 生成汉明矩阵、标签矩阵
    gt_martix = np.dot(test_labels, np.transpose(labels))  # 100*4000
    logging.info('{}gt_martix finish!'.format(getNowTime()))
    ham_matrix = np.dot(test_codes, np.transpose(codes))  # hanmming distance map to dot value 100*4000
    logging.info('{}ham_martix finish!'.format(getNowTime()))

    # 汉明矩阵排序
    sorted_ham_martix = np.argsort(ham_matrix, axis=1)  #
    # logging.info(sorted_ham_martix)
    # logging.info(ham_martix[0][sorted_ham_martix[0]])
    # 计算准确率
    logging.info('{}sort ham_matrix finished,start calculate precision'.format(getNowTime()))

    p = np.zeros((numOfTest, 1), np.float64)  # 100*1

#####################################
    for i in range(numOfTest):
        n = 0.0
        test_matrix = sorted_ham_martix[i, :]
        length = test_matrix.shape[0]
        #logging.info('第%d个查询为:%s' % (i + 1, Names[i]))
       # logging.info('\t与第%d个样本汉明距离从小到大排序的top-%d是：'%(1+i, k))
        for j in range(k):
            if gt_martix[i][test_matrix[length - j - 1]] == 1:
                x[int(lab[i])] += 1  # 正样本个数
                n += 1
        # logging.info('\t%s'%Names[test_matrix[length - j - 1]])
        count[int(lab[i])] +=1
       # print(count)
        p[i] = n / k
   #     logging.info('precision of top-k(k=%d) is:%f' % (k, np.mean(p)))
    logging.info('mean precision by top-k(k=%d) is:%4f'%(k, np.mean(p)))
    for i in range(Classes):
        acc[i] = x[i] /  (count[i]*k)
        logging.info('第%d类的top %d准确率为：%4f'%(i, k, acc[i]))
    end = time.time()
    logging.info('Average evaluate time is %4f s' % ((end - start) / numOfTest))

    precision = np.mean(acc)
    logging.info('mean precision by top-k(k=%d) is:%4f'%(k, precision))
    return precision


#计算查全率
def Calculate_precision(HASHING_BITS):
    start = time.time()
    x = np.zeros((Classes, 1), np.int32)
    acc = np.zeros((Classes, 1), np.float64)
    count = np.zeros((Classes, 1), np.float64)
    ##############################

    train_codes = []  # 训练集二进制码
    train_labels = []  # 训练集标签数组
    test_codes = []  # 测试集二进制码
    test_labels = []  # 测试集标签数组
    codes = []  # 全部数据集二进制码
    labels = []  # 全部数据级标签数组
    lab = []  # 标签值
    Names = []  # 数据集图片名
    Get_Names(Names)
    # get g.t. and binary code
    Get_CodesLabels(train_codes, train_labels, test_codes, test_labels, lab,HASHING_BITS)
    codes.extend(test_codes)
    codes.extend(train_codes)
    labels.extend(test_labels)
    labels.extend(train_labels)
    codes = np.array(codes)
    labels = np.array(labels)
    train_codes = np.array(train_codes)  # 4000*24
    train_labels = np.array(train_labels)  # 4000*5
    test_codes = np.array(test_codes)  # 100*24
    test_labels = np.array(test_labels)  # 100*5

    # 生成汉明矩阵、标签矩阵
    gt_martix = np.dot(test_labels, np.transpose(labels))  # 100*4000
    logging.info('{}gt_martix finish!'.format(getNowTime()))
    ham_matrix = np.dot(test_codes, np.transpose(codes))  # hanmming distance map to dot value 100*4000
    logging.info('{}ham_martix finish!'.format(getNowTime()))

    # 汉明矩阵排序
    sorted_ham_martix = np.argsort(ham_matrix, axis=1)  #
    # logging.info(sorted_ham_martix)
    # logging.info(ham_martix[0][sorted_ham_martix[0]])
    # 计算准确率
    logging.info('{}sort ham_matrix finished,start calculate precision'.format(getNowTime()))

    p = np.zeros((numOfTest, 1), np.float64)  # 100*1

####################################
    logging.info('\n=================================4 计算查全率================================')
    for i in range(numOfTest):
        xx = 0.0
        test_matrix = sorted_ham_martix[i, :]
        length = test_matrix.shape[0]
        for j in range(return_num):
            if gt_martix[i][test_matrix[length - j - 1]] == 1:
                x[int(lab[i])] += 1  #正样本个数
                xx += 1
        count[int(lab[i])] += 1
        p[i] = xx / return_num
    logging.info('平均查全率 is:%4f' % np.mean(p))
    for i in range(Classes):
        acc[i] = x[i] / (return_num * count[i])
        logging.info('第%d类的查全率为：%4f' % (i, acc[i]))
    end = time.time()
    logging.info('Average time is %4f s' % ((end - start) / numOfTest))
    precision = np.mean(acc)
    logging.info('平均查全率 is:%4f'%precision)
    #return precision


def precisionALL(HASHING_BITS):

    timestr = strftime("Start time :%Y-%m-%d %H:%M:%S")

    logging.info(timestr)
    # 计算准确率
    Calculate_ByMinHanmming(HASHING_BITS )
    Calculate_ByHanmming_d(HASHING_BITS )
    k=5
    Calculate_Top_k(k,HASHING_BITS )
    k = 10
    Calculate_Top_k(k, HASHING_BITS)

    k =20
    Calculate_Top_k(k, HASHING_BITS)

    k = 40
    Calculate_Top_k(k, HASHING_BITS)
   # k = 50
  #  Calculate_Top_k(k, HASHING_BITS)

    k = 60
    Calculate_Top_k(k, HASHING_BITS)
    k = 80
    Calculate_Top_k(k, HASHING_BITS)

    k = 100
    Calculate_Top_k(k, HASHING_BITS)
 #   k = 120
  #  Calculate_Top_k(k, HASHING_BITS)

  #  k =150
  #  Calculate_Top_k(k, HASHING_BITS)

    Calculate_precision(HASHING_BITS )
    logging.info('{}evaluate finished'.format(getNowTime()))

    '''
 
#计算准确率
if __name__ == '__main__':
    print('{}start!'.format(getNowTime()))

    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/Logs/'
    log_name = log_path + rq  +"-bits"+ str(HASHING_BITS) +"k" + str(k)+ '.log'

    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)

    timestr = strftime("Start time :%Y-%m-%d %H:%M:%S")

    logging.info(timestr)

    train_codes = []  #训练集二进制码
    train_labels = []  #训练集标签数组
    test_codes = []  #测试集二进制码
    test_labels = []  #测试集标签数组
    codes = [] #全部数据集二进制码
    labels = [] #全部数据级标签数组
    lab = []  #标签值
    Names = [] #数据集图片名
    Get_Names(Names)
    # get g.t. and binary code
    Get_CodesLabels(train_codes, train_labels, test_codes, test_labels, lab)
    codes.extend(test_codes)
    codes.extend(train_codes)
    labels.extend(test_labels)
    labels.extend(train_labels)
    codes = np.array(codes)
    labels = np.array(labels)
    train_codes = np.array(train_codes)  # 4000*24
    train_labels = np.array(train_labels)  # 4000*5
    test_codes = np.array(test_codes)  # 100*24
    test_labels = np.array(test_labels)  # 100*5

    # 生成汉明矩阵、标签矩阵
    gt_martix = np.dot(test_labels, np.transpose(labels))  # 100*4000
    logging.info('{}gt_martix finish!'.format(getNowTime()))
    ham_matrix = np.dot(test_codes, np.transpose(codes))  # hanmming distance map to dot value 100*4000
    logging.info('{}ham_martix finish!'.format(getNowTime()))

    # 汉明矩阵排序
    sorted_ham_martix = np.argsort(ham_matrix, axis=1)  #
    #logging.info(sorted_ham_martix)
    #logging.info(ham_martix[0][sorted_ham_martix[0]])
    # 计算准确率
    logging.info('{}sort ham_matrix finished,start calculate precision'.format(getNowTime()))

    p = np.zeros((numOfTest, 1), np.float64)  # 100*1

	#计算准确率
    Calculate_ByMinHanmming()
    Calculate_ByHanmming_d()
    Calculate_Top_k(5)
    Calculate_Top_k(10)
    Calculate_Top_k(20)
    Calculate_Top_k(50)
    Calculate_Top_k(100)
    Calculate_Top_k(150)
    Calculate_precision()
    logging.info('{}evaluate finished'.format(getNowTime()))
     '''

