
import numpy as np
from time import strftime, gmtime
import  Build_Code
import Precision
import time
import os
import mAP
import Build_Code
import logging
TEST_NUM = 100
TOP_NUM = 400 # 计算top 400
Test_count = 100  #测试集大小
Train_count = 4000  #训练集大小
Classes = 5   #类别数
numOfTest = 100  #测试图片数
numOfAll = 4100  #全部图片数
#k = 60   #top-k的k 10 20 40 60 100 150
d = 5 #指定汉明距离
return_num = 200  #召回数

def TestTRAIN_MAP(HASHING_BITS,   alpha,regularizer  ):
    timestr = strftime("Start time :%Y-%m-%d %H:%M:%S")
    logging.info("BitLength:%i" % (HASHING_BITS))
    logging.info(timestr)
	#训练模型e
    start = time.time()
 #   print("HASHING_BITS:",HASHING_BITS)
    print("Training,HASHING_BITS:%i" % HASHING_BITS)

    Build_Code.train(HASHING_BITS, alpha,regularizer)
    end = time.time()
    logging.info("Training time: %f s" % (end - start))
    print("Train end")
  #######################################
	#计算二进制编码
    start = time.time()
    print("Encoding.......................")
    Build_Code.evaluate_Codes(HASHING_BITS)
    end = time.time()
    print("Encode end.......................")
    logging.info("Coding time: %f s" % (end - start))

    timestr = strftime("Evaluate Codes End time :%Y-%m-%d %H:%M:%S")
    logging.info(timestr)

    #############################################################
    print("Precision ......................")
    Precision.precisionALL(HASHING_BITS)

    print("Precision end.......................")
############################MAP#################################
    print("MAP.......................")
    mAP.Evaluate_mAP_YPS(HASHING_BITS)
    print("Map End.......................")
    timestr = strftime("Map End. time :%Y-%m-%d %H:%M:%S")
    logging.info(timestr)
    Build_Code.reset()

if __name__ == '__main__':
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/Logs/'

    log_name = log_path + rq + "ALL" + '.log'
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    #############################
    #  alpha = 0.01, 0, 0, 001, 0, 01, 0, 1

    alpha = 0.1
    regularizer = True  ######L1 regularizer
    logging.info("===============regularizer L1, alpha%f======================="%alpha)
    HASHING_BITS = 8
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 16
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 24
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 32
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 48
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 64
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)

    alpha = 0.01
    regularizer = True ######L1 regularizer
    logging.info("===============regularizer L1, alpha%f=======================" % alpha)
    HASHING_BITS = 8
    TestTRAIN_MAP(HASHING_BITS,alpha,regularizer)
    HASHING_BITS = 16
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 24
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 32
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 48
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 64
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)

    alpha = 0.001
    regularizer = True ######L1 regularizer
    logging.info("===============regularizer L1, alpha%f=======================" % alpha)
    HASHING_BITS = 8
    TestTRAIN_MAP(HASHING_BITS,alpha,regularizer)
    HASHING_BITS = 16
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 24
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 32
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 48
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 64
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)

    alpha = 0.0001
    regularizer = True  ######L1 regularizer
    logging.info("===============regularizer L1, alpha%f=======================" % alpha)
    HASHING_BITS = 8
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 16
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 24
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 32
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 48
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 64
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)


#########################L2 regularizer#############################
    alpha = 0.1
    regularizer = False  ######L2 regularizer
    logging.info("===============regularizer L2, alpha%f=======================" % alpha)
    HASHING_BITS = 8
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 16
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 24
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 32
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 48
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 64
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)

    alpha = 0.01
    logging.info("===============regularizer L2, alpha%f=======================" % alpha)
    regularizer = False  ######L2 regularizer
    HASHING_BITS = 8
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 16
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 24
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 32
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 48
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 64
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)

    alpha = 0.001
    regularizer = False  ######L2 regularizer
    logging.info("===============regularizer L2, alpha%f=======================" % alpha)
    HASHING_BITS = 8
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 16
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 24
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 32
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 48
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 64
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)


    alpha = 0.0001
    regularizer = False  ######L2 regularizer
    logging.info("===============regularizer L2, alpha%f=======================" % alpha)
    HASHING_BITS = 8
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 16
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 24
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 32
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 48
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)
    HASHING_BITS = 64
    TestTRAIN_MAP(HASHING_BITS, alpha, regularizer)



