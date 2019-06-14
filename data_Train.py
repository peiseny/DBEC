#!/usr/bin/python
# -*- coding: utf-8 -*-import os
import numpy as np
from PIL import Image

import pickle as pk

import os


imgSize = 64  #图片大小
data = {}
list1 = []
list2 = []
list3 = []
CURRENT_DIR = os.getcwd()  # 获取当前路径


# 将测试集图像信息写进二进制文件
def convert_trainbin():
    global data
    global list1
    global list2
    global list3

    for k in range(0, num):
        # 按名读取图片
        currentpath = folder + "/" + imgName[k]
        im = Image.open(currentpath)
        # img = tf.gfile.FastGFile(currentpath, 'rb').read()
        # image = tf.image.decode_jpeg(img)
        # adjust = tf.image.per_image_standardization(image)  #标准化图像
        # 获取RGB值
        with open(binpath, 'a') as f:
            for i in range(0, imgSize):
                for j in range(0, imgSize):
                    cl = im.getpixel((i, j))
                    list1.append(cl[0])
            for i in range(0, imgSize):
                for j in range(0, imgSize):
                    cl = im.getpixel((i, j))
                    list1.append(cl[1])

            for i in range(0, imgSize):
                for j in range(0, imgSize):
                    cl = im.getpixel((i, j))
                    list1.append(cl[2])
        list2.append(list1)
        list1 = []
        f.close()
        print("{} saved.".format(imgName[k]))

        list3.append(imgName[k])
        # list3.append(imgName[k].encode('utf-8'))

    arr2 = np.array(list2, dtype=np.uint8)

    # 字典形式存储图像信息
    data['batch_label'] = 'training batch'
    data.setdefault('labels', label)
    data.setdefault('data', arr2)
    data.setdefault('filenames', list3)

    # data['batch_label'.encode('utf-8')]='testing batch 1 of 1'.encode('utf-8')
    # data.setdefault('labels'.encode('utf-8'),label)
    # data.setdefault('data'.encode('utf-8'),arr2)
    # data.setdefault('filenames'.encode('utf-8'),list3)
    output = open(binpath, 'wb')
    pk.dump(data, output)  # 序列化对象，并将结果数据流写入到文件对象中
    output.close()

imgName = []  # 存放测试集图片名的数组
label = []  # 存放测试集图片标签的数组
# 从txt文件读图片名和标签
with open('juhua/new_trainLabels.txt', 'r') as f:
    lines = f.readlines()
    index = 0
    for line in lines:
        filename, filelabel = line.rstrip().split(' ')
        index += 1
        print('[{}]:{}:{}'.format(index,filename,filelabel))
        imgName.append(filename)
        label.append(int(filelabel))
# print(label)

num = len(imgName)  # 测试集图片个数
# print(imgNum)
folder = 'juhua/train'
binpath = 'juhua/train_batch'
convert_trainbin()

