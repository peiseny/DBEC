import numpy as np
import time
import os

import logging
from time import strftime, gmtime
import time


# read train and test binaryCodes
CURRENT_DIR = os.getcwd()
TEST_NUM = 100
TOP_NUM = 400 # 计算top 400

def Get_Time():
    '''
    获取当前时间
    '''
    return '[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ']'


def Get_HammingDistance(a, b):
    '''
    计算汉明距离，对于二进制串ａ和ｂ来说，汉明距离等于 a XOR b 中 1 的数目
    '''
    distance = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            distance += 1
    return distance


# 从txt读取Binary Codes和GroundTruth
def Get_BinaryCode(train_codes, train_groudTruth, test_codes, test_groudTruth,hashing_bits):
    '''
    获取result.txt文件中测试和训练结果
    '''
    file_res_name = CURRENT_DIR + "/result" + str(hashing_bits) + ".txt"
    number = 0
    with open(file_res_name, 'r') as f:
        for line in f:
            temp = line.strip().split('\t')
            # 前3600条为测试数据，后面是训练数据
            if number < TEST_NUM:
                list2 = [0] * 32
                # list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                list2[int(temp[1])] = 1  # 标签对应位改为1，oneshot
                test_groudTruth.append(list2)  # get test ground truth(0-9)
                test_codes.append([i if i == 1
                                    else -1 for i in map(int, list(temp[0]))])  # list生成，读取每一行的hashcode，1保留，0改为-1
            else:
                list2 = [0] * 32
                list2[int(temp[1])] = 1
                train_groudTruth.append(list2)  # get test ground truth(0-9)
                train_codes.append([i if i == 1
                                    else -1 for i in map(int, list(temp[0]))])  # 1保留，0改为-1

            number += 1
    print('{}read data finish'.format(Get_Time()))


# 从txt读取Binary Codes和GroundTruth
def Get_BinaryCode_YPS(train_codes, hashing_bits,  train_groudTruth, test_codes, test_groudTruth): ### yps 20180622
    '''
    获取result.txt文件中测试和训练结果
    '''


    file_res_name = CURRENT_DIR + "/result" + str(hashing_bits) + ".txt"
	
    print("Get_BinaryCode_YPS", hashing_bits)
    number = 0
    with open(file_res_name, 'r') as f:
        for line in f:
            temp = line.strip().split('\t')
            # 前3600条为测试数据，后面是训练数据
            # if number < 10000:
            if number < TEST_NUM:
                list2 = [0] * 32
                # list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                list2[int(temp[1])] = 1  # 标签对应位改为1，oneshot
                test_groudTruth.append(list2)  # get test ground truth(0-9)
                test_codes.append([i if i == 1
                                    else -1 for i in map(int, list(temp[0]))])  # list生成，读取每一行的hashcode，1保留，0改为-1
            else:
                list2 = [0] * 32
                list2[int(temp[1])] = 1
                train_groudTruth.append(list2)  # get test ground truth(0-9)
                train_codes.append([i if i == 1
                                    else -1 for i in map(int, list(temp[0]))])  # 1保留，0改为-1
            number += 1
    print('{}read data finish'.format(Get_Time()))



def Evaluate_mAP_YPS(hashing_bits ):   ### yps 20180622
    logging.info('{}start!'.format(Get_Time()))
    logging.info("Evaluate_mAP_YPS:%i"%(hashing_bits) )
    test_codes = []
    train_codes = []
    test_groudTruth = []
    train_groudTruth = []

    # get g.t. and binary code
    Get_BinaryCode_YPS(train_codes,hashing_bits, train_groudTruth, test_codes, test_groudTruth)
    test_codes = np.array(test_codes)  # 3600*12
    test_groudTruth = np.array(test_groudTruth)  # 3600*32
    train_codes = np.array(train_codes)  # 10000*12
    train_groudTruth = np.array(train_groudTruth)  # 10000*32

    gt_matrix = np.dot(test_groudTruth, np.transpose(train_groudTruth))  # 矩阵乘法 3600*10000
    logging.info('{}gt_matrix finish!'.format(Get_Time()))
    ham_matrix = np.dot(test_codes, np.transpose(train_codes))  # hanmming distance map to dot value 3600*10000
    logging.info('{}ham_matrix finish!'.format(Get_Time()))

    sorted_ham_matrix = np.argsort(ham_matrix, axis=1)  # 按行从小到大排序
    logging.info(sorted_ham_matrix)
    logging.info(ham_matrix[0][sorted_ham_matrix[0]])
    # calculate mAP
    logging.info('{}sort ham_matrix finished,start calculate mAP'.format(Get_Time()))

    AP_all = np.zeros((TEST_NUM, 1), np.float64)  # 3600*1
    start = time.time()
    for i in range(TEST_NUM):
        x = 0.0
        p = 0
        test_oneLine = sorted_ham_matrix[i, :]
        length = test_oneLine.shape[0]

        for j in range(TOP_NUM):  # 从后往前计算，即从大到小计算
            if gt_matrix[i][test_oneLine[length - j - 1]] == 1:  # reverse
                x += 1  # 选中的正样本个数
                p += x / (j + 1)  # precision=检测正确的数据/总的检测个数
        if p == 0:
            AP_all[i] = 0
        else:
            AP_all[i] = p / x  # AP值

    end = time.time()
    logging.info("Average time:%f s" % ((end - start) / TEST_NUM))
    mAP = np.mean(AP_all)
    logging.info('{}mAP:{}'.format(Get_Time(), mAP))




    '''
if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/Logs/'
    log_name = log_path + rq + "--"+str(TOP_NUM) +'MAP' + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)



    logging.info('{}start!'.format(Get_Time()))

    Evaluate_mAP_YPS(8)
    '''
################################################################################
    '''
    test_codes = []
    train_codes = []
    test_groudTruth = []
    train_groudTruth = []

    # get g.t. and binary code
    Get_BinaryCode(train_codes,  train_groudTruth, test_codes, test_groudTruth)
    test_codes = np.array(test_codes)  # 3600*12
    test_groudTruth = np.array(test_groudTruth)  # 3600*32
    train_codes = np.array(train_codes)  # 10000*12
    train_groudTruth = np.array(train_groudTruth)  # 10000*32

    gt_matrix = np.dot(test_groudTruth, np.transpose(train_groudTruth))  # 矩阵乘法 3600*10000
    logging.info('{}gt_matrix finish!'.format(Get_Time()))
    ham_matrix = np.dot(test_codes, np.transpose(train_codes))  # hanmming distance map to dot value 3600*10000
    logging.info('{}ham_matrix finish!'.format(Get_Time()))

    sorted_ham_matrix = np.argsort(ham_matrix, axis=1)  # 按行从小到大排序
    logging.info(sorted_ham_matrix)
    logging.info(ham_matrix[0][sorted_ham_matrix[0]])
    # calculate mAP
    logging.info('{}sort ham_matrix finished,start calculate mAP'.format(Get_Time()))

    AP_all = np.zeros((TEST_NUM, 1), np.float64)  # 3600*1
    start = time.time()
    for i in range(TEST_NUM):
        x = 0.0
        p = 0
        test_oneLine = sorted_ham_matrix[i, :]
        length = test_oneLine.shape[0]

        for j in range(TOP_NUM):  # 从后往前计算，即从大到小计算
            if gt_matrix[i][test_oneLine[length - j - 1]] == 1:  # reverse
                x += 1  # 选中的正样本个数
                p += x / (j + 1)  # precision=检测正确的数据/总的检测个数
        if p == 0:
            AP_all[i] = 0
        else:
            AP_all[i] = p / x  # AP值

            end = time.time()
    logging.info("Average time:%f s" % ((end - start) / TEST_NUM))
    mAP = np.mean(AP_all)
    logging.info('{}mAP:{}'.format(Get_Time(), mAP))


    
    '''


